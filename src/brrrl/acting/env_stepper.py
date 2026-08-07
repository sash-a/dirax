from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

type StepResult[Obs] = tuple[
    Obs,
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.bool_],
    dict[str, Any],
]


class EnvStepper[Obs, Act](Protocol):
    """The gymnasium step API, with a leading lane axis on every field."""

    def reset(self) -> tuple[Obs, dict[str, Any]]: ...

    def step(self, action: Act) -> StepResult[Obs]: ...


@dataclass
class GymStepper[Obs, Act]:
    """One `gym.Env` presented as a single lane: ``N = 1``."""

    env: gym.Env

    def reset(self) -> tuple[Obs, dict[str, Any]]:
        raise NotImplementedError

    def step(self, action: Act) -> StepResult[Obs]:
        raise NotImplementedError
