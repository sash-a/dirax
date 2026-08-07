from dataclasses import dataclass

import optax
from echo import TrajectoryAccumulator
from numpy.typing import NDArray

from brrrl.acting.env_stepper import EnvStepper
from brrrl.acting.policy import Policy
from brrrl.algorithms.ppo.types import PPOLearnerState, PPOOutput, PPOTrajectory
from brrrl.types import PRNGKey, Metrics


@dataclass
class PPORoller[Obs]:
    stepper: EnvStepper[Obs, NDArray]
    policy: Policy[Obs, PPOOutput]
    accumulator: TrajectoryAccumulator
    num_steps: int
    key: PRNGKey

    def rollout(self) -> tuple[PPOTrajectory, Metrics]:
        raise NotImplementedError


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
