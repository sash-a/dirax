from collections.abc import Callable

import numpy as np
from echo import TrajectoryAccumulator
from numpy.typing import ArrayLike, NDArray

from brrrl.acting.env_stepper import EnvStepper, finished_episodes
from brrrl.acting.policy import Policy, PolicyOutput
from brrrl.types import TRANSITIONS, VERSION, Metrics, Trajectory, Transition

type RecordFn[Obs, O: PolicyOutput] = Callable[
    [Obs, NDArray[np.bool_], NDArray[np.bool_], NDArray[np.floating], O], Transition
]


class Roller[Obs, O: PolicyOutput, T: Trajectory]:
    """The one rollout loop, shared by every algorithm.

    It owns the autoreset discipline, the trailing slot, the Version stamp and the episode
    statistics, so none of those can drift between algorithms. What an algorithm brings is a
    `record` function and the `trajectory` type to rebuild at the end, and nothing else.

    A Policy adopts new params at whatever rate it likes, so the stamp is the mean of the Versions
    its acts actually ran under rather than one taken on the way in.

    Its environments are reset once, here, so consecutive Trajectories are one unbroken stream of
    interaction. The Trajectory it returns aliases the accumulator's buffers, so a caller finishes
    with one before asking for the next.
    """

    def __init__(
        self,
        stepper: EnvStepper[Obs, ArrayLike],
        policy: Policy[Obs, O],
        accumulator: TrajectoryAccumulator,
        record: RecordFn[Obs, O],
        trajectory: type[T],
        num_steps: int,
    ) -> None:
        self.stepper = stepper
        self.policy = policy
        self.accumulator = accumulator
        self.record = record
        self.trajectory = trajectory
        self.num_steps = num_steps
        self.prev_obs, _ = stepper.reset()
        self.prev_term = np.zeros(stepper.num_envs, np.bool_)
        self.prev_trunc = np.zeros(stepper.num_envs, np.bool_)

    @property
    def num_transitions(self) -> int:
        """How many Transitions one rollout produces. The trailing slot is never stepped, so it is
        not one of them."""
        return self.num_steps * self.stepper.num_envs

    def rollout(self) -> tuple[T, Metrics]:
        infos, versions = [], []
        for _ in range(self.num_steps):
            output, version = self.policy.act(self.prev_obs)
            observation, reward, terminated, truncated, info = self.stepper.step(output.action)
            transition = self.record(self.prev_obs, self.prev_term, self.prev_trunc, reward, output)
            self.accumulator.add(TRANSITIONS, transition)
            self.prev_obs, self.prev_term, self.prev_trunc = observation, terminated, truncated
            infos.append(info)
            versions.append(version)

        trailing, version = self.policy.act(self.prev_obs)
        no_reward = np.zeros(self.stepper.num_envs)
        transition = self.record(self.prev_obs, self.prev_term, self.prev_trunc, no_reward, trailing)
        self.accumulator.add(TRANSITIONS, transition)
        versions.append(version)

        self.accumulator.add(VERSION, np.array([np.mean(versions)], np.float32))
        return self.trajectory(**self.accumulator.build()), finished_episodes(infos)
