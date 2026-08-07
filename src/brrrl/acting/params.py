from dataclasses import dataclass
from typing import Protocol

from brrrl.types import Params, Version, VersionedParams


class ParamSource(Protocol):
    @property
    def current(self) -> VersionedParams: ...


class ParamSink(Protocol):
    def publish(self, params: Params, version: Version) -> None: ...


@dataclass
class InMemoryParamStore:
    params: VersionedParams

    @property
    def current(self) -> VersionedParams:
        raise NotImplementedError

    def publish(self, params: Params, version: Version) -> None:
        raise NotImplementedError
