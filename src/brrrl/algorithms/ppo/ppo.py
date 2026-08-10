from dataclasses import dataclass

import optax

from brrrl.algorithms.ppo.types import PPOLearnerState, PPOTrajectory
from brrrl.types import Metrics, PRNGKey


@dataclass
class PPOLearner:
    optimizer: optax.GradientTransformation
    clip_eps: float
    num_epochs: int

    def init(self, key: PRNGKey) -> PPOLearnerState:
        raise NotImplementedError

    def update(self, state: PPOLearnerState, trajectory: PPOTrajectory) -> tuple[PPOLearnerState, Metrics]:
        raise NotImplementedError


def main() -> None:
    """Train in one process, one thread, no ray and no echo transport."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
