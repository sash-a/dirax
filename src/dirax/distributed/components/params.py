import asyncio
from typing import ClassVar, Literal, overload

import ray
from ray import ObjectRef
from ray.actor import ActorProxy

from dirax.distributed.placement.component import Behaviour
from dirax.types import Params, VersionedParams
from dirax.utils.duration import Duration, EveryTicks, Schedule


class RayParamStore:
    """The one place live params exist; the Learner writes, Policies read.

    Empty until the Learner's opening publish, because params are built where they are trained
    rather than shipped from the driver.
    """

    RAY_OPTIONS: ClassVar[Behaviour] = {"max_restarts": 0}

    def __init__(self, reader_ids: list[int]) -> None:
        self.latest: VersionedParams | None = None
        self._fresh: dict[int, asyncio.Queue[None]] = {reader_id: asyncio.Queue(1) for reader_id in reader_ids}

    @ray.method
    async def publish(self, params: VersionedParams) -> None:
        """Takes the Version it is given: the writer mints them, so no reader has to ask for one."""
        self.latest = params
        for queue in self._fresh.values():
            if queue.empty():  # A reader two publishes behind still only needs waking once.
                queue.put_nowait(None)

    @ray.method
    async def get(self, reader_id: int) -> VersionedParams:
        """Parks until a publish this reader has not been woken for lands."""
        await self._fresh[reader_id].get()
        assert self.latest is not None  # A wake is only ever put after a publish has set this.
        return self.latest

    @ray.method
    def version(self) -> int:
        return self.latest.version if self.latest is not None else 0


class RayParamSink:
    """The only writer into a ``RayParamStore``, and so the run's version authority."""

    def __init__(self, store: ActorProxy[RayParamStore], every: Duration | None = None, version: int = -1) -> None:
        """
        The writer into the ``RayParamStore``

        Args:
            store: the remote store where params are kept.
            every: how frequently to push params, defaults to every call to `publish`.
            version: the Version the last publish carried. Left at -1 a fresh run's opening publish
              is Version 0; a resumed run states what its checkpoint left off at.
        """
        self.store = store
        self.every = Schedule(every or EveryTicks(1))
        self.version = version

    def publish(self, params: Params) -> int:
        self.version += 1
        if self.every.due():
            self.store.publish.remote(VersionedParams(params, self.version))

        return self.version


class RayParamSource:
    """A reader's half of the long-poll: one request always in flight, claimed when it lands."""

    def __init__(self, store: ActorProxy[RayParamStore], id_ref: int) -> None:
        self.store = store
        self.id_ref = id_ref
        self.param_request: list[ObjectRef] = [self.store.get.remote(id_ref)]

    @overload
    def get(self, block: Literal[True]) -> VersionedParams: ...
    @overload
    def get(self, block: Literal[False]) -> VersionedParams | None: ...
    def get(self, block: bool) -> VersionedParams | None:
        """Costs no round trip: a request is always already in flight, so this only asks if it landed."""
        ready_params, _ = ray.wait(self.param_request, num_returns=1, timeout=None if block else 0)
        if not ready_params:
            return None

        self.param_request = [self.store.get.remote(self.id_ref)]  # Re-armed before the deserialise.
        return ray.get(ready_params[0])
