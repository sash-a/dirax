import time

import numpy as np
from echo import TrajectoryAccumulator
from numpy.typing import ArrayLike

from brrrl.acting.env_stepper import EnvStepper
from brrrl.acting.policy import Policy, PolicyOutput
from brrrl.metrics.timer import AggregatedElapsed, Timer
from brrrl.types import TRANSITIONS, ArrayTree, Metrics, PolicyInput, Rollout, Trajectory, Transition
from brrrl.wrappers.statistics import env_stats, episode_stats


class Roller[Obs: ArrayTree, I: PolicyInput, O: PolicyOutput, Tran: Transition, Traj: Trajectory]:
    """The one rollout loop, shared by every algorithm.

    It owns the autoreset discipline, the bootstrapped slot, the parameter Version and the episode
    statistics, so none of those can drift between algorithms. What an algorithm brings is a stepper,
    policy, `trajectory` and `transition` types and nothing else.
    """

    def __init__(
        self,
        stepper: EnvStepper[Obs, ArrayLike],
        policy: Policy[I, O],
        accumulator: TrajectoryAccumulator,
        policy_input: type[I],
        transition: type[Tran],
        trajectory: type[Traj],
        num_steps: int,
        seed: int,
    ) -> None:
        self.stepper = stepper
        self.policy = policy
        self.accumulator = accumulator
        self.trajectory = trajectory
        self.transition = transition
        self.policy_input = policy_input
        self.num_steps = num_steps
        self.width = (stepper.num_envs, stepper.num_agents) if stepper.num_agents > 1 else (stepper.num_envs,)
        self.timer = Timer(AggregatedElapsed)

        # initial step
        self.prev_obs, _ = stepper.reset(seed=seed)
        self.prev_term = np.zeros(self.width, np.bool_)
        self.prev_trunc = np.zeros(self.width, np.bool_)
        policy_input = self.policy_input.of(self.prev_obs, self.prev_term, self.prev_trunc)
        self.decision, self.version = self.policy.act(policy_input)

    @property
    def num_transitions(self) -> int:
        """How many Transitions one rollout produces."""
        return self.num_steps * self.stepper.num_envs

    def close(self) -> None:
        """Releases the environments."""
        self.stepper.close()

    def rollout(self) -> tuple[Rollout[Traj], Metrics]:
        infos, versions = [], []
        start_time = time.monotonic()
        for _ in range(self.num_steps):
            with self.timer("env_step_time"):
                obs, reward, term, trunc, info = self.stepper.step(self.decision.action)
            with self.timer("env_step_bookkeeping_time"):
                transition = self.transition.record(
                    self.prev_obs, self.prev_term, self.prev_trunc, reward, self.decision
                )
                self.accumulator.add(TRANSITIONS, transition)
                self.prev_obs, self.prev_term, self.prev_trunc = obs, term, trunc
                infos.append(info)
                versions.append(self.version)
            with self.timer("policy_act_time"):
                self.decision, self.version = self.policy.act(self.policy_input.of(obs, term, trunc))

        with self.timer("episode_bookkeeping_time"):
            no_reward = np.zeros(self.width)
            transition = self.transition.record(
                self.prev_obs, self.prev_term, self.prev_trunc, no_reward, self.decision
            )
            self.accumulator.add(TRANSITIONS, transition)
            version = np.array([np.mean(versions)], np.float32)
            batch = Rollout(self.trajectory(**self.accumulator.build()), version)

        sps = self.num_steps / (time.monotonic() - start_time)
        metrics = {**episode_stats(infos), **env_stats(infos), **self.timer.dump(), "acted_steps_per_second": sps}
        return batch, metrics
