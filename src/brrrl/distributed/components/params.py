import asyncio
from typing import ClassVar, Literal, overload

import ray
from ray import ObjectRef
from ray.actor import ActorProxy

from brrrl.distributed.placement.component import Behaviour
from brrrl.params.params import ParamStore
from brrrl.types import Params, VersionedParams
from brrrl.utils.duration import Duration, EveryTicks, Schedule


class RayParamStore:
    """The one place live params exist; the Learner writes, Policies read.

    Holds the in-process Store rather than reimplementing it, and hands each reader its own
    edge-triggered wake.
    """

    RAY_OPTIONS: ClassVar[Behaviour] = {"max_restarts": 0}

    def __init__(self, store: ParamStore, reader_ids: list[int]) -> None:
        self.store = store
        self._fresh: dict[int, asyncio.Queue[None]] = {reader_id: asyncio.Queue(1) for reader_id in reader_ids}
        for queue in self._fresh.values():
            queue.put_nowait(None)  # The driver seeded the Store, so no reader waits for its first params.

    @ray.method
    async def publish(self, params: Params) -> int:
        version = self.store.publish(params)
        for queue in self._fresh.values():
            if queue.empty():  # A reader two publishes behind still only needs waking once.
                queue.put_nowait(None)
        return version

    @ray.method
    async def version(self) -> int:
        """The current Version, for a caller that wants it without spending a reader's wake."""
        return self.store.get(block=True).version

    @ray.method
    async def get(self, reader_id: int) -> VersionedParams:
        """Parks until a publish this reader has not been woken for lands."""
        await self._fresh[reader_id].get()
        return self.store.get(block=True)


class RayParamSink:
    """The writer into the ``RayParamStore``

    Duplicated the version that lives in the store that we don't need to wait for a round trip to
    return the version when sending params. Only right while it is the store's only writer.
    """

    def __init__(self, store: ActorProxy[RayParamStore], every: Duration | None = None) -> None:
        """
        The writer into the ``RayParamStore``

        Args:
            store: the remote store where params are kept.
            every: how frequently to push params, defaults to every call to `publish`.
        """
        self.store = store
        self.every = Schedule(every or EveryTicks(1))
        self.version = ray.get(self.store.version.remote())

    def publish(self, params: Params) -> int:
        if self.every.due():
            self.store.publish.remote(params)
            self.version += 1

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
