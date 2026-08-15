from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import concatenate, create_empty_array
from numpy.typing import NDArray

from brrrl.types import Metrics

type StepResult[Obs] = tuple[
    Obs,
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.bool_],
    dict[str, Any],
]

EPISODE_STATS = "episode"
EPISODE_MASK = "_episode"
EPISODE_METRICS = {"r": "episode_return", "l": "episode_length", "t": "episode_time"}
"""The vector episode-statistics schema, and what each of its fields is reported as."""

OWNED_WRAPPERS = (gym.wrappers.Autoreset, gym.wrappers.RecordEpisodeStatistics)
"""What a stepper applies itself, and so must not be handed a second copy of."""


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

    Wraps the provided env in an autoreset and metrics wrapper for bookkeeping.
    """

    def __init__(self, env: gym.Env) -> None:
        if owned := [wrapper.__name__ for wrapper in applied(env) if issubclass(wrapper, OWNED_WRAPPERS)]:
            raise ValueError(
                f"{owned} already wraps this env, and ``GymStepper`` applies that itself. Pass the env without it."
            )
        self.env = gym.wrappers.Autoreset(gym.wrappers.RecordEpisodeStatistics(env))

    @property
    def num_envs(self) -> int:
        return 1

    def reset(self) -> tuple[Obs, dict[str, Any]]:
        observation, info = self.env.reset()
        return self.batched(observation), info

    def step(self, action: Act) -> StepResult[Obs]:
        observation, reward, terminated, truncated, info = self.env.step(np.asarray(action)[0])
        return (
            self.batched(observation),
            np.array([reward]),
            np.array([terminated]),
            np.array([truncated]),
            widened(info),
        )

    def batched(self, observation: Any) -> Any:
        """A leading env axis of one, in a fresh array: a Roller carries an observation across a step,
        so a buffer reused between steps would rewrite the slot it already recorded.

        The observation space decides the structure, which is what ``Obs`` is the caller's claim about.
        """
        space = self.env.observation_space
        return concatenate(space, [observation], create_empty_array(space, n=1))


class GymVectorStepper[Obs, Act]:
    """``num_envs`` copies of one `gym.Env` presented as an `EnvStepper`, batching every act call.

    It vectorises what it is handed rather than taking an already-vectorised env, so the autoreset
    mode the Trajectory layout rests on is this class's decision and not a caller's (ADR-0001).
    That is why the constructor takes a factory: `num_envs` distinct envs cannot be made from one
    instance.
    """

    def __init__(self, make_env: Callable[[], gym.Env], num_envs: int) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs is {num_envs}, and a stepper with no environments steps nothing.")
        self.env = gym.wrappers.vector.RecordEpisodeStatistics(
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
    """``make_env``, refusing anything a vector stepper would end up wrapping twice.

    The vector env owns autoreset and the statistics wrapper across all of its envs, so a per-env
    copy of either is the same double-wrap `GymStepper` refuses, one level down.
    """

    def build() -> gym.Env:
        env = make_env()
        if isinstance(env, gym.vector.VectorEnv):
            raise TypeError(
                f"{type(env).__name__} is already vectorised, and ``GymVectorStepper`` vectorises what it "
                "is given. Pass a factory for one env and the count you want."
            )
        if owned := [wrapper.__name__ for wrapper in applied(env) if issubclass(wrapper, OWNED_WRAPPERS)]:
            raise ValueError(
                f"{owned} already wraps this env, and a stepper applies that itself. Pass the env without it."
            )
        return env

    return build


def widened(info: dict[str, Any]) -> dict[str, Any]:
    """The single-env statistics wrapper reports scalars and no mask, so this is what a stepper owes
    a Roller for `finished_episodes` to stay the only place that knows the schema.
    """
    if EPISODE_STATS not in info:
        return info
    stats = info[EPISODE_STATS]
    return info | {
        EPISODE_STATS: {key: np.array([stats[key]]) for key in EPISODE_METRICS},
        EPISODE_MASK: np.ones(1, np.bool_),
    }


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
