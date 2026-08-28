from dataclasses import dataclass
from typing import ClassVar, Literal, overload

from ray.actor import ActorHandle

from brrrl.distributed.placement.component import Behaviour
from brrrl.params.params import ParamStore
from brrrl.types import Params, VersionedParams


class RayParamStore:
    """The one place live params exist; the Learner writes, Policies read.

    Holds the in-process Store rather than reimplementing it. The edge-triggered long-poll that
    keeps a reader from ever blocking lands with the node-local Mirror, which is a Source too.
    """

    RAY_OPTIONS: ClassVar[Behaviour] = {"max_restarts": 0}

    def __init__(self, store: ParamStore) -> None:
        self.store = store

    def publish(self, params: Params) -> int:
        return self.store.publish(params)

    def get(self, block: bool) -> VersionedParams | None:
        return self.store.get(block)


@dataclass
class RayParamSink:
    store: ActorHandle[RayParamStore]

    def publish(self, params: Params) -> int:
        raise NotImplementedError


@dataclass
class RayParamSource:
    store: ActorHandle[RayParamStore]

    @overload
    def get(self, block: Literal[True]) -> VersionedParams: ...
    @overload
    def get(self, block: Literal[False]) -> VersionedParams | None: ...
    def get(self, block: bool) -> VersionedParams | None:
        raise NotImplementedError
