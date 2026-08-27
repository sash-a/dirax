from dataclasses import dataclass
from typing import Literal, Protocol, overload

from brrrl.types import Params, Version, VersionedParams


class ParamSource(Protocol):
    """The read half. Identical for a local Store and a remote Mirror, so the links compose."""

    @overload
    def get(self, block: Literal[True]) -> VersionedParams: ...
    @overload
    def get(self, block: Literal[False]) -> VersionedParams | None: ...
    def get(self, block: bool) -> VersionedParams | None:
        """The latest params, or None where nothing new has landed and the caller would not wait.

        Stated at every call, because it is the return type that turns on it rather than a default
        anyone would want.
        """
        ...


class ParamSink(Protocol):
    def publish(self, params: Params) -> Version:
        """Store ``params`` under the next Version, and report which one that was."""
        ...


@dataclass
class ParamStore:
    """The one place live params exist in a process. Seeded by the driver, so no reader waits.

    Its Version is the run's monotonic authority: a resumed run seeds the count it left off at.
    """

    params: VersionedParams

    def get(self, block: bool) -> VersionedParams:
        """Never None: nothing can land while a caller in this process waits, so nothing is waited on."""
        return self.params

    def publish(self, params: Params) -> Version:
        self.params = VersionedParams(params, self.params.version + 1)
        return self.params.version
