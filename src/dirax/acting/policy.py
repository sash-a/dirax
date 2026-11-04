from collections.abc import Callable
from typing import Any, Protocol, final

import jax

from dirax.params.params import ParamSource
from dirax.types import ArrayTree, Params, PolicyInput, PolicyOutput, PRNGKey

# jittable acting functions
type ActFn[I: PolicyInput, O: PolicyOutput] = Callable[[Params, I, PRNGKey], O]
type RecActFn[I: PolicyInput, O: PolicyOutput] = Callable[[Params, I, PRNGKey, ArrayTree], tuple[O, ArrayTree]]


class Policy[I: PolicyInput, O: PolicyOutput](Protocol):
    def act(self, policy_input: I) -> tuple[O, int]:
        """Acting function"""
        ...

    def state(self) -> Any:
        """Allows callers to obtain a hidden state."""
        ...


@final
class FFPolicy[I: PolicyInput, O: PolicyOutput]:
    """Owns its params and its key, so a Roller holds no jax state and no param concept.

    Its first params are pulled on construction rather than at the first Sync point, so the wait
    for a Learner that has not published yet is paid once, here, and a Sync point never blocks.
    """

    def __init__(self, act_fn: ActFn[I, O], source: ParamSource, key: PRNGKey) -> None:
        self.act_fn = act_fn
        self.source = source
        self.key = key
        self.params = source.get(block=True)

    def sync(self) -> int:
        """Never blocks, allows for speed at the cost of staleness."""
        if (latest := self.source.get(block=False)) is not None:
            self.params = latest
        return self.params.version

    def act(self, policy_input: I) -> tuple[O, int]:
        """Syncs first, so a publish reaches a rollout the step after it lands rather than the next
        Trajectory. One device-to-host transfer for the whole tree, not one per leaf per step."""
        version = self.sync()
        self.key, key = jax.random.split(self.key)
        return jax.device_get(self.act_fn(self.params.value, policy_input, key)), version

    def state(self) -> None:
        """Feedforward policies have no hidden state."""


@final
class RecPolicy[I: PolicyInput, O: PolicyOutput]:
    """An `FFPolicy` that also carries hidden state, which stays on the device between acts.

    It always holds the hidden state it's next ``act`` will use.
    """

    def __init__(self, act_fn: RecActFn[I, O], carry: ArrayTree, source: ParamSource, key: PRNGKey) -> None:
        self.act_fn = act_fn
        self.carry = carry
        self.source = source
        self.key = key
        self.params = source.get(block=True)

    def sync(self) -> int:
        """Never blocks, allows for speed at the cost of staleness."""
        if (latest := self.source.get(block=False)) is not None:
            self.params = latest
        return self.params.version

    def act(self, policy_input: I) -> tuple[O, int]:
        """Only the output crosses to the host; the carry stays where the next act will want it."""
        version = self.sync()
        self.key, key = jax.random.split(self.key)
        output, self.carry = self.act_fn(self.params.value, policy_input, key, self.carry)
        return jax.device_get(output), version

    def state(self) -> ArrayTree:
        return self.carry
