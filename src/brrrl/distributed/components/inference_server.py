from dataclasses import dataclass
from typing import ClassVar

import ray
from ray.actor import ActorProxy

from brrrl.acting.policy import LocalPolicy, PolicyOutput
from brrrl.distributed.placement.component import Behaviour


class InferenceServer[Obs, O: PolicyOutput]:
    """A `LocalPolicy` behind RPC; refreshes params on its own cadence."""

    RAY_OPTIONS: ClassVar[Behaviour] = {}

    def __init__(self, policy: LocalPolicy[Obs, O]) -> None:
        raise NotImplementedError

    @ray.method
    def version(self) -> int:
        raise NotImplementedError

    @ray.method
    def act(self, observations: Obs) -> tuple[O, int]:
        """Keys are the server's own: batching across Actors already makes such a run non-reproducible."""
        raise NotImplementedError


@dataclass
class InferenceServerPolicy[Obs, O: PolicyOutput]:
    """Client half. Coalesces concurrent env requests into one batched call."""

    server: ActorProxy[InferenceServer[Obs, O]]

    def act(self, observation: Obs) -> tuple[O, int]:
        """The Version rides back on a call the client was making anyway, so aligning costs nothing."""
        raise NotImplementedError
