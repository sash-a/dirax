from typing import Protocol

from brrrl.types import PRNGKey, LearnerState, Metrics, Trajectory


class Learner[S: LearnerState, T: Trajectory](Protocol):
    def init(self, key: PRNGKey) -> S: ...

    def update(self, state: S, trajectory: T) -> tuple[S, Metrics]: ...
