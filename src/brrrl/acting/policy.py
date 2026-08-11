from collections.abc import Callable
from typing import Protocol

import jax

from brrrl.params.params import ParamSource
from brrrl.types import Array, Params, PRNGKey, Version


class PolicyOutput(Protocol):
    @property
    def action(self) -> Array: ...


type ActFn[Obs, O: PolicyOutput] = Callable[[Params, Obs, PRNGKey], O]
"""The jittable half, where params and a key stay explicit because a Policy is what supplies them."""


class Policy[Obs, O: PolicyOutput](Protocol):
    def act(self, observation: Obs) -> tuple[O, Version]:
        """Act on host arrays and return them, beside the Version the params behind them carry.

        When to align with new params is the Policy's own business, so nothing here forces a Sync
        point — and a remote Policy reports its Version on a call it was making anyway.
        """
        ...


class LocalPolicy[Obs, O: PolicyOutput]:
    """Owns its params and its key, so a Roller holds no jax state and no param concept.

    Its first params are pulled on construction rather than at the first Sync point, because the
    driver has already seeded the Store — which is what lets a Sync point never block.
    """

    def __init__(self, act_fn: ActFn[Obs, O], source: ParamSource, key: PRNGKey) -> None:
        self.act_fn = act_fn
        self.source = source
        self.key = key
        self.params = source.get(block=True)

    def sync(self) -> Version:
        """Never blocks: staleness has no reason to become a stall."""
        if (latest := self.source.get(block=False)) is not None:
            self.params = latest
        return self.params.version

    def act(self, observation: Obs) -> tuple[O, Version]:
        """Syncs first, so a publish reaches a rollout the step after it lands rather than the next
        Trajectory. One device-to-host transfer for the whole tree, not one per leaf per step."""
        version = self.sync()
        self.key, key = jax.random.split(self.key)
        return jax.device_get(self.act_fn(self.params.value, observation, key)), version
