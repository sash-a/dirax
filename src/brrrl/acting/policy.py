from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from numpy.typing import NDArray

from brrrl.acting.params import ParamSource
from brrrl.types import PRNGKey, Params, Version, VersionedParams


class PolicyOutput(Protocol):
    @property
    def action(self) -> NDArray: ...


type ActFn[Obs, O: PolicyOutput] = Callable[[Params, Obs, PRNGKey], O]


class Policy[Obs, O: PolicyOutput](Protocol):
    def sync(self) -> Version:
        """Align with the latest params and report the version now being acted under."""
        ...

    def act(self, observation: Obs, key: PRNGKey) -> O: ...


@dataclass
class LocalPolicy[Obs, O: PolicyOutput]:
    act_fn: ActFn[Obs, O]
    source: ParamSource
    params: VersionedParams | None = None

    def sync(self) -> Version:
        raise NotImplementedError

    def act(self, observation: Obs, key: PRNGKey) -> O:
        raise NotImplementedError
