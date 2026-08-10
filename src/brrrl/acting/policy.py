from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import jax

from brrrl.acting.params import ParamSource
from brrrl.types import Array, Params, PRNGKey, Version, VersionedParams


class PolicyOutput(Protocol):
    @property
    def action(self) -> Array:
        """Either residency: an act function builds this on the device, a Roller reads it on the host."""
        ...


type ActFn[Obs, O: PolicyOutput] = Callable[[Params, Obs, PRNGKey], O]


class Policy[Obs, O: PolicyOutput](Protocol):
    def sync(self) -> Version:
        """Align with the latest params and report the version now being acted under."""
        ...

    def act(self, observation: Obs) -> O:
        """Act on host arrays and return them, holding no key and no params of the caller's."""
        ...


@dataclass
class LocalPolicy[Obs, O: PolicyOutput]:
    """Owns its params and its keys, so a Roller holds no jax state and no param concept."""

    act_fn: ActFn[Obs, O]
    source: ParamSource
    key: PRNGKey
    keys_per_refill: int = 512
    params: VersionedParams = field(init=False)
    keys: PRNGKey = field(init=False)
    used: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.params = self.source.get(block=True)
        self.refill()

    def sync(self) -> Version:
        """Never blocks: the driver seeded the Store, so staleness has no reason to become a stall."""
        if (latest := self.source.get()) is not None:
            self.params = latest
        return self.params.version

    def act(self, observation: Obs) -> O:
        """One device-to-host transfer for the whole tree, rather than one per leaf per step."""
        return jax.device_get(self.act_fn(self.params.value, observation, self.next_key()))

    def next_key(self) -> PRNGKey:
        if self.used == len(self.keys):
            self.refill()
        key = self.keys[self.used]
        self.used += 1
        return key

    def refill(self) -> None:
        """One split per refill rather than one per step, the first key carrying on as the next seed."""
        keys = jax.random.split(self.key, self.keys_per_refill + 1)
        self.key, self.keys, self.used = keys[0], keys[1:], 0
