from collections.abc import Callable
from typing import Protocol

import jax

from brrrl.params.params import ParamSource
from brrrl.types import Array, ArrayTree, Params, PolicyOutput, PRNGKey

type ActFn[Obs, O: PolicyOutput] = Callable[[Params, Obs, PRNGKey], O]

# Input: params, obs, key, hstate, done
# Output: policy output, hstate
type RecActFn[Obs, O: PolicyOutput] = Callable[[Params, Obs, PRNGKey, ArrayTree, Array], tuple[O, ArrayTree]]


class Policy[Obs, O: PolicyOutput](Protocol):
    def act(self, observation: Obs, done: Array) -> tuple[O, int]:
        """Act on observations and return actions, beside the Version of the params."""
        ...


class FFPolicy[Obs, O: PolicyOutput]:
    """A feed-forward policy. It pulls new params every step if available.

    Its first params are pulled on construction rather than at the first Sync point, so the wait
    for a Learner that has not published yet is paid once, here, and a Sync point never blocks.
    """

    def __init__(self, act_fn: ActFn[Obs, O], source: ParamSource, key: PRNGKey) -> None:
        self.act_fn = act_fn
        self.source = source
        self.key = key
        self.params = source.get(block=True)

    def sync(self) -> int:
        """Never blocks, allows for speed at the cost of staleness."""
        if (latest := self.source.get(block=False)) is not None:
            self.params = latest
        return self.params.version

    def act(self, observation: Obs, done: Array) -> tuple[O, int]:
        """Syncs first, so a publish reaches a rollout the step after it lands rather than the next Trajectory."""
        version = self.sync()
        self.key, key = jax.random.split(self.key)
        return jax.device_get(self.act_fn(self.params.value, observation, key)), version


class RecPolicy[Obs, O: PolicyOutput]:
    """A recurrent-forward policy with a hidden-state. It pulls new params every step if available."""

    def __init__(self, act_fn: RecActFn[Obs, O], init_hstate: ArrayTree, source: ParamSource, key: PRNGKey) -> None:
        self.act_fn = act_fn
        self.hstate = init_hstate
        self.source = source
        self.key = key
        self.params = source.get(block=True)

    def sync(self) -> int:
        if (latest := self.source.get(block=False)) is not None:
            self.params = latest
        return self.params.version

    def act(self, observation: Obs, done: Array) -> tuple[O, int]:
        version = self.sync()
        self.key, key = jax.random.split(self.key)
        policy_output, self.hstate = self.act_fn(self.params.value, observation, key, self.hstate, done)
        # TODO: we might want to avoid the overhead of D2H/network transfer for hstates
        #  some algorithms work with using the initial hstates or first from the rollout.
        #  It's quite hard to do this and keep things clean/understandable
        return jax.device_get(policy_output), version
