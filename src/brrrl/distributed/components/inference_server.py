from dataclasses import dataclass
from typing import ClassVar

from ray.actor import ActorHandle

from brrrl.acting.policy import LocalPolicy, PolicyOutput
from brrrl.distributed.placement.component import Behaviour
from brrrl.types import Version


class InferenceServer[Obs, O: PolicyOutput]:
    """A `LocalPolicy` behind RPC; refreshes params on its own cadence."""

    RAY_OPTIONS: ClassVar[Behaviour] = {}

    def __init__(self, policy: LocalPolicy[Obs, O]) -> None:
        raise NotImplementedError

    def version(self) -> Version:
        raise NotImplementedError

    def act(self, observations: Obs) -> O:
        """Keys are the server's own: batching across Actors already makes such a run non-reproducible."""
        raise NotImplementedError


@dataclass
class InferenceServerPolicy[Obs, O: PolicyOutput]:
    """Client half. Coalesces concurrent env requests into one batched call."""

    server: ActorHandle

    def sync(self) -> Version:
        raise NotImplementedError

    def act(self, observation: Obs) -> O:
        raise NotImplementedError
