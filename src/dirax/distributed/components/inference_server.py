from dataclasses import dataclass
from typing import Any, ClassVar

import ray
from ray.actor import ActorProxy

from dirax.acting.policy import FFPolicy, PolicyInput, PolicyOutput
from dirax.distributed.placement.component import Behaviour


class InferenceServer[I: PolicyInput, O: PolicyOutput]:
    """A `FFPolicy` behind RPC; refreshes params on its own cadence."""

    RAY_OPTIONS: ClassVar[Behaviour] = {}

    def __init__(self, policy: FFPolicy[I, O]) -> None:
        raise NotImplementedError

    @ray.method
    def version(self) -> int:
        raise NotImplementedError

    @ray.method
    def act(self, policy_input: I) -> tuple[O, int]:
        """Keys are the server's own: batching across Actors already makes such a run non-reproducible."""
        raise NotImplementedError


@dataclass
class InferenceServerPolicy[I: PolicyInput, O: PolicyOutput]:
    """Client half. Coalesces concurrent env requests into one batched call."""

    server: ActorProxy[InferenceServer[I, O]]

    def act(self, policy_input: I) -> tuple[O, int]:
        """The Version rides back on a call the client was making anyway, so aligning costs nothing."""
        raise NotImplementedError

    def state(self) -> Any:
        """Whatever the served policy carries between acts, for a Roller to record."""
        raise NotImplementedError
