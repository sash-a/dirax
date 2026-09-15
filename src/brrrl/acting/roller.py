import time
from collections.abc import Callable

import numpy as np
from echo import TrajectoryAccumulator
from numpy.typing import ArrayLike, NDArray

from brrrl.acting.env_stepper import EnvStepper
from brrrl.acting.policy import Policy, PolicyOutput
from brrrl.metrics.timer import AggregatedElapsed, Timer
from brrrl.types import TRANSITIONS, Metrics, Rollout, Trajectory, Transition
from brrrl.wrappers.statistics import env_stats, episode_stats

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
        seed: int,
    ) -> None:
        """The shape every field leads with comes off the stepper, which is what knows it: ``(E, A)``
        for a team and ``(E,)`` for a single agent, which has no agent axis at all. This is the only
        place a Roller asks how many agents there are."""
        self.stepper = stepper
        self.policy = policy
        self.accumulator = accumulator
        self.record = record
        self.trajectory = trajectory
        self.num_steps = num_steps
        self.width = (stepper.num_envs, stepper.num_agents) if stepper.num_agents > 1 else (stepper.num_envs,)
        self.prev_obs, _ = stepper.reset(seed=seed)
        self.prev_term = np.zeros(self.width, np.bool_)
        self.prev_trunc = np.zeros(self.width, np.bool_)
        self.timer = Timer(AggregatedElapsed)

    @property
    def num_transitions(self) -> int:
        """How many Transitions one rollout produces. A Transition is one environment step however
        many agents took it, so this does not multiply by the agent count. The trailing slot is never
        stepped, so it is not one of them."""
        return self.num_steps * self.stepper.num_envs

    def close(self) -> None:
        """Releases the environments. The Roller built none of them, but it is the only thing that
        holds the stepper, so it is the only place a caller can reach them from."""
        self.stepper.close()

    def rollout(self) -> tuple[Rollout[T], Metrics]:
        infos, versions = [], []
        start_time = time.monotonic()
        for _ in range(self.num_steps):
            with self.timer("policy_act_time"):
                output, version = self.policy.act(self.prev_obs)
            with self.timer("env_step_time"):
                observation, reward, terminated, truncated, info = self.stepper.step(output.action)
            with self.timer("env_step_bookkeeping_time"):
                transition = self.record(self.prev_obs, self.prev_term, self.prev_trunc, reward, output)
                self.accumulator.add(TRANSITIONS, transition)
                self.prev_obs, self.prev_term, self.prev_trunc = observation, terminated, truncated
                infos.append(info)
                versions.append(version)

        with self.timer("policy_act_time"):
            trailing, version = self.policy.act(self.prev_obs)

        with self.timer("episode_bookkeeping_time"):
            no_reward = np.zeros(self.width)
            transition = self.record(self.prev_obs, self.prev_term, self.prev_trunc, no_reward, trailing)
            self.accumulator.add(TRANSITIONS, transition)
            versions.append(version)
            version = np.array([np.mean(versions)], np.float32)
            batch = Rollout(self.trajectory(**self.accumulator.build()), version)

        sps = self.num_steps / (time.monotonic() - start_time)
        metrics = {**episode_stats(infos), **env_stats(infos), **self.timer.dump(), "acted_steps_per_second": sps}
        return batch, metrics
