from typing import Protocol

from brrrl.types import Metrics, Trajectory


class Roller[T: Trajectory](Protocol):
    def rollout(self) -> tuple[T, Metrics]: ...
