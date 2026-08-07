from dataclasses import dataclass

import ray
from ray.actor import ActorHandle

from brrrl.types import Params, Version, VersionedParams


@ray.remote
class ParamStore:
    """The one place live params exist; learner writes, policies read."""

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
