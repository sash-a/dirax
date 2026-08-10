from collections.abc import Iterable, Mapping
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode
from gymnasium.wrappers.vector import RecordEpisodeStatistics
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


class GymVectorStepper[Obs, Act]:
    """A gymnasium vector env presented as an `EnvStepper`. One env is ``num_envs=1``."""

    def __init__(self, env: gym.vector.VectorEnv) -> None:
        mode = env.metadata.get("autoreset_mode")
        if mode is not AutoresetMode.NEXT_STEP:
            raise ValueError(
                f"autoreset mode is {mode}, and the Trajectory layout is only valid under "
                f"{AutoresetMode.NEXT_STEP}: any other mode puts an observation in a slot whose "
                f"flags describe a different episode. See ADR-0001."
            )
        self.env = RecordEpisodeStatistics(env)

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def reset(self) -> tuple[Obs, dict[str, Any]]:
        return self.env.reset()

    def step(self, action: Act) -> StepResult[Obs]:
        return self.env.step(action)


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
