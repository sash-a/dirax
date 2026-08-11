from dataclasses import dataclass
from typing import ClassVar, Literal, overload

from ray.actor import ActorHandle

from brrrl.distributed.placement.component import Behaviour
from brrrl.params import InMemoryParamStore
from brrrl.types import Params, Version, VersionedParams


class ParamStore:
    """The one place live params exist; the Learner writes, Policies read.

    Holds the in-process Store rather than reimplementing it. The edge-triggered long-poll that
    keeps a reader from ever blocking lands with the node-local Mirror, which is a Source too.
    """

    RAY_OPTIONS: ClassVar[Behaviour] = {"max_restarts": 0}

    def __init__(self, store: InMemoryParamStore) -> None:
        self.store = store

    def publish(self, params: Params) -> Version:
        return self.store.publish(params)

    def get(self, block: bool) -> VersionedParams | None:
        return self.store.get(block)


@dataclass
class RayParamSink:
    store: ActorHandle

    def publish(self, params: Params) -> Version:
        raise NotImplementedError


@dataclass
class RayParamSource:
    store: ActorHandle

    @overload
    def get(self, block: Literal[True]) -> VersionedParams: ...
    @overload
    def get(self, block: Literal[False]) -> VersionedParams | None: ...
    def get(self, block: bool) -> VersionedParams | None:
        raise NotImplementedError
