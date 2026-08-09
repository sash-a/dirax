from dataclasses import dataclass
from typing import ClassVar

from ray.actor import ActorHandle

from brrrl.distributed.placement.component import Behaviour
from brrrl.types import Params, Version, VersionedParams


class ParamStore:
    """The one place live params exist; learner writes, policies read."""

    RAY_OPTIONS: ClassVar[Behaviour] = {"max_restarts": 0}
    """A restarted store comes back empty, and every Policy then acts on stale params without an error."""

    def publish(self, params: Params, version: Version) -> None:
        raise NotImplementedError

    def current(self) -> VersionedParams:
        raise NotImplementedError


@dataclass
class RayParamSink:
    store: ActorHandle

    def publish(self, params: Params, version: Version) -> None:
        raise NotImplementedError


@dataclass
class RayParamSource:
    store: ActorHandle

    @property
    def current(self) -> VersionedParams:
        raise NotImplementedError
