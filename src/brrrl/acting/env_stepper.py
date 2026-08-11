from collections.abc import Iterable, Mapping
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

    It implements next-step autoreset itself rather than asserting one, so the step whose action
    the env ignores is the step it resets on. The env behind the statistics wrapper therefore never
    receives that step, which is why its episode statistics are exact with no previous-dones
    bookkeeping — the same property an own vectoriser would inherit.
    """

    def __init__(self, env: gym.Env) -> None:
        self.env = gym.wrappers.RecordEpisodeStatistics(env)
        self.done = False

    @property
    def num_envs(self) -> int:
        return 1

    def reset(self) -> tuple[Obs, dict[str, Any]]:
        observation, info = self.env.reset()
        self.done = False
        return self.batched(observation), info

    def step(self, action: Act) -> StepResult[Obs]:
        if self.done:
            return self.autoreset()
        observation, reward, terminated, truncated, info = self.env.step(np.asarray(action)[0])
        self.done = terminated or truncated
        return (
            self.batched(observation),
            np.array([reward], np.float64),
            np.array([terminated]),
            np.array([truncated]),
            widened(info),
        )

    def autoreset(self) -> StepResult[Obs]:
        """The step the env ignores: a new episode's first observation, no reward and no flags."""
        observation, _ = self.env.reset()
        self.done = False
        return self.batched(observation), np.zeros(1), np.zeros(1, np.bool_), np.zeros(1, np.bool_), {}

    def batched(self, observation: Any) -> Any:
        """A leading env axis of one, in a fresh array: a Roller carries an observation across a step,
        so a buffer reused between steps would rewrite the slot it already recorded.

        The observation space decides the structure, which is what ``Obs`` is the caller's claim about.
        """
        space = self.env.observation_space
        return concatenate(space, [observation], create_empty_array(space, n=1))


class GymVectorStepper[Obs, Act]:
    """A gymnasium vector env presented as an `EnvStepper`, batching every act call across its envs.

    Gymnasium owns the autoreset here, so this one asserts the mode rather than implementing it.
    """

    def __init__(self, env: gym.vector.VectorEnv) -> None:
        mode = env.metadata.get("autoreset_mode")
        if mode is not AutoresetMode.NEXT_STEP:
            raise ValueError(
                f"autoreset mode is {mode}, and the Trajectory layout is only valid under "
                f"{AutoresetMode.NEXT_STEP}: any other mode puts an observation in a slot whose "
                f"flags describe a different episode. See ADR-0001."
            )
        self.env = gym.wrappers.vector.RecordEpisodeStatistics(env)

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def reset(self) -> tuple[Obs, dict[str, Any]]:
        return self.env.reset()

    def step(self, action: Act) -> StepResult[Obs]:
        return self.env.step(action)


def widened(info: dict[str, Any]) -> dict[str, Any]:
    """One env's episode statistics in the vector schema: full-width fields, and the mask naming it.

    The single-env statistics wrapper reports scalars and no mask, so this is what a stepper owes a
    Roller for `finished_episodes` to stay the only place that knows the schema.
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
