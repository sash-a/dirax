from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from echo import TrajectoryAccumulator
from numpy.typing import ArrayLike, NDArray

from brrrl.acting.env_stepper import EnvStepper, finished_episodes
from brrrl.acting.policy import Policy, PolicyOutput
from brrrl.types import TRANSITIONS, VERSION, Metrics, Trajectory, Transition


class Roller[T: Trajectory](Protocol):
    @property
    def num_transitions(self) -> int:
        """How many Transitions one rollout produces, which only a Roller knows both factors of."""
        ...

    def rollout(self) -> tuple[T, Metrics]: ...


type RecordFn[Obs, O: PolicyOutput] = Callable[
    [Obs, NDArray[np.bool_], NDArray[np.bool_], NDArray[np.floating], O], Transition
]
"""A slot, in the order it aligns: observation, terminated, truncated, reward, policy output."""


@dataclass
class Rollout[Obs, O: PolicyOutput, T: Trajectory]:
    """The one rollout loop, shared by every algorithm and jax-free.

    It owns the autoreset discipline, the trailing slot, the Version stamp and the episode
    statistics, so none of those can drift between algorithms. What an algorithm brings is a
    `record` function and the `trajectory` type to rebuild at the end, and nothing else.

    The returned Trajectory aliases the accumulator's buffers, so a caller finishes with one
    before asking for the next.
    """

    stepper: EnvStepper[Obs, ArrayLike]
    policy: Policy[Obs, O]
    accumulator: TrajectoryAccumulator
    record: RecordFn[Obs, O]
    trajectory: type[T]
    num_steps: int
    observation: Obs = field(init=False)
    terminated: NDArray[np.bool_] = field(init=False)
    truncated: NDArray[np.bool_] = field(init=False)

    def __post_init__(self) -> None:
        """The one reset a Roller does: from here the stream is unbroken across Trajectories."""
        self.observation, _ = self.stepper.reset()
        self.terminated = np.zeros(self.stepper.num_envs, np.bool_)
        self.truncated = np.zeros(self.stepper.num_envs, np.bool_)

    @property
    def num_transitions(self) -> int:
        """The trailing slot is never stepped, so it is not one of them."""
        return self.num_steps * self.stepper.num_envs

    def rollout(self) -> tuple[T, Metrics]:
        version = self.policy.sync()
        self.accumulator.add(VERSION, np.full(1, version, np.int32))

        infos = []
        for _ in range(self.num_steps):
            output = self.policy.act(self.observation)
            observation, reward, terminated, truncated, info = self.stepper.step(output.action)
            self.accumulator.add(TRANSITIONS, self.slot(reward, output))
            self.observation, self.terminated, self.truncated = observation, terminated, truncated
            infos.append(info)

        trailing = self.policy.act(self.observation)
        no_reward = np.zeros(self.stepper.num_envs)
        self.accumulator.add(TRANSITIONS, self.slot(no_reward, trailing))
        return self.trajectory(**self.accumulator.build()), finished_episodes(infos)

    def slot(self, reward: NDArray[np.floating], output: O) -> Transition:
        """Every slot is written by the algorithm's own record function, the trailing one included."""
        return self.record(self.observation, self.terminated, self.truncated, reward, output)
