from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv

from brrrl.types import Metrics, StepResult
from brrrl.wrappers.widening import EPISODE_MASK, EPISODE_STATS, Widened

EPISODE_METRICS = {"r": "episode_return", "l": "episode_length", "t": "episode_time"}
"""What each field of the episode-statistics schema is reported as."""

OWNED_WRAPPERS = (gym.wrappers.Autoreset,)
"""What a stepper applies itself, and so must not be handed a second copy of."""

REWARD_WRAPPERS = (gym.RewardWrapper, gym.wrappers.NormalizeReward)
"""What scales a reward, and so what episode statistics have to be taken underneath."""

type EnvWrap = Callable[[gym.Env], gym.Env]
type VectorWrap = Callable[[VectorEnv], VectorEnv]
"""What a stepper hands its envs to once it has applied its own. A stepper knows nothing of what
either does, only that its statistics are underneath and its autoreset above."""


def same[E](env: E) -> E:
    """The wrap a stepper takes when a caller names none."""
    return env


class EnvStepper[Obs, Act](Protocol):
    """The gymnasium step API, with a leading env axis on every field.

    Beside the step it reports its finished episodes in the vector episode-statistics schema:
    full-width ``r``, ``l`` and ``t`` under ``episode``, and the boolean mask naming which envs
    finished under ``_episode``. That contract is what lets a Roller read episodes the same way
    whatever built the environments.
    """

    @property
    def num_envs(self) -> int: ...

    def reset(self) -> tuple[Obs, dict[str, Any]]: ...

    def step(self, action: Act) -> StepResult[Obs]: ...


class GymStepper[Obs, Act]:
    """One `gym.Env` presented as an `EnvStepper`: ``num_envs=1``.

    Applies the autoreset the Trajectory layout rests on, the episode statistics its half of the
    `EnvStepper` contract needs, and the widening that gives every field the env axis a Roller
    reads. Everything else about an env is the caller's, which is what leaves this holding only the
    count and the refusal.
    """

    def __init__(self, env: gym.Env, wrap: EnvWrap = same) -> None:
        """``wrap`` is the single-env half of the vector stepper's: applied over the statistics, so
        what it scales is what a Learner sees and never what an episode return is reported in."""
        self.env = Widened(gym.wrappers.Autoreset(wrap(recorded(steppable(env)))))

    @property
    def num_envs(self) -> int:
        return 1

    def reset(self) -> tuple[Obs, dict[str, Any]]:
        return self.env.reset()

    def step(self, action: Act) -> StepResult[Obs]:
        return self.env.step(action)


class GymVectorStepper[Obs, Act]:
    """``num_envs`` copies of one `gym.Env` presented as an `EnvStepper`, batching every act call.

    It vectorises what it is handed rather than taking an already-vectorised env, so the autoreset
    mode the Trajectory layout rests on is this class's decision and not a caller's (ADR-0001).
    That is why the constructor takes a factory: `num_envs` distinct envs cannot be made from one
    instance.
    """

    def __init__(self, make_env: Callable[[], gym.Env], num_envs: int, wrap: VectorWrap = same) -> None:
        """Vectorise ``make_env`` ``num_envs`` times, and hand the result to ``wrap``.

        Args:
            make_env: Builds one env.
            num_envs: How many to build.
            wrap: Applied to the vectorised envs, for a transform that wants them all at once.
        """
        if num_envs < 1:
            raise ValueError(f"num_envs is {num_envs}, and a stepper with no environments steps nothing.")
        self.env = wrap(
            gym.vector.SyncVectorEnv([vectorisable(make_env)] * num_envs, autoreset_mode=AutoresetMode.NEXT_STEP)
        )

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def reset(self) -> tuple[Obs, dict[str, Any]]:
        return self.env.reset()

    def step(self, action: Act) -> StepResult[Obs]:
        return self.env.step(action)


def applied(env: gym.Env) -> Iterator[type[gym.Wrapper]]:
    """Every wrapper already around ``env``, outermost first."""
    while isinstance(env, gym.Wrapper):
        yield type(env)
        env = env.env


def vectorisable(make_env: Callable[[], gym.Env]) -> Callable[[], gym.Env]:
    """``make_env``, refusing anything a vector stepper cannot step and recording what it must.

    The vector env owns the autoreset across all of its envs, so a per-env copy of it is the same
    double-wrap `GymStepper` refuses, one level down.
    """

    def build() -> gym.Env:
        env = make_env()
        if isinstance(env, gym.vector.VectorEnv):
            raise TypeError(
                f"{type(env).__name__} is already vectorised, and ``GymVectorStepper`` vectorises what it "
                "is given. Pass a factory for one env and the count you want."
            )
        return recorded(steppable(env))

    return build


def steppable(env: gym.Env) -> gym.Env:
    """``env``, refusing a wrapper a stepper applies itself and would otherwise apply twice."""
    if owned := [wrapper.__name__ for wrapper in applied(env) if issubclass(wrapper, OWNED_WRAPPERS)]:
        raise ValueError(f"{owned} already wraps this env, and a stepper applies that itself. Pass the env without it.")
    return env


def recorded(env: gym.Env) -> gym.Env:
    """``env`` reporting its finished episodes, in the units the environment itself pays out in.

    A stepper applies the statistics when nothing has scaled the reward yet, which is most runs.
    Once something has, only the caller can place them: they have to sit under the scaling, and a
    wrapper applied from here would sit above it — reporting every episode return in units that
    move with the policy, which is not something a later reader can undo or even notice.
    """
    chain = list(applied(env))  # Outermost first, so a wrapper deeper in it is one applied earlier.
    statistics = [i for i, wrapper in enumerate(chain) if issubclass(wrapper, gym.wrappers.RecordEpisodeStatistics)]
    scaling = [i for i, wrapper in enumerate(chain) if issubclass(wrapper, REWARD_WRAPPERS)]
    if not scaling:
        return env if statistics else gym.wrappers.RecordEpisodeStatistics(env)
    if not statistics or min(statistics) < max(scaling):
        raise ValueError(
            f"{[chain[i].__name__ for i in scaling]} scales this env's reward, so "
            f"{gym.wrappers.RecordEpisodeStatistics.__name__} has to wrap it underneath and be applied by you. "
            "Otherwise every episode return is logged in scaled units."
        )
    return env


def finished_episodes(infos: Iterable[Mapping[str, Any]]) -> Metrics:
    """Every episode that finished over a rollout, read out of the statistics schema.

    A rollout in which none finished reports no keys at all rather than a zero or a nan. A stepper
    that reports statistics without the mask naming which envs they belong to fails here, rather
    than reporting nothing for the rest of the run.
    """
    finished = [(info[EPISODE_STATS], info[EPISODE_MASK]) for info in infos if EPISODE_STATS in info]
    if not finished:
        return {}
    return {
        name: np.concatenate([stats[key][mask] for stats, mask in finished]) for key, name in EPISODE_METRICS.items()
    }
