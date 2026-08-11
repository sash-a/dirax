from collections.abc import Callable
from typing import Protocol

import jax

from brrrl.params import ParamSource
from brrrl.types import Array, Params, PRNGKey, Version


class PolicyOutput(Protocol):
    @property
    def action(self) -> Array: ...


type ActFn[Obs, O: PolicyOutput] = Callable[[Params, Obs, PRNGKey], O]
"""The jittable half, where params and a key stay explicit because a Policy is what supplies them."""


class Policy[Obs, O: PolicyOutput](Protocol):
    def sync(self) -> Version:
        """Align with the latest params and report the version now being acted under."""
        ...

    def act(self, observation: Obs) -> O:
        """Act on host arrays and return them, holding no key and no params of the caller's."""
        ...


class LocalPolicy[Obs, O: PolicyOutput]:
    """Owns its params and its keys, so a Roller holds no jax state and no param concept.

    Its first params are pulled on construction rather than at the first Sync point, because the
    driver has already seeded the Store — which is what lets a Sync point never block.
    """

    def __init__(self, act_fn: ActFn[Obs, O], source: ParamSource, key: PRNGKey, keys_per_refill: int = 512) -> None:
        self.act_fn = act_fn
        self.source = source
        self.key = key
        self.keys_per_refill = keys_per_refill
        self.params = source.get(block=True)
        self.refill()

    def sync(self) -> Version:
        """Never blocks: staleness has no reason to become a stall."""
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
