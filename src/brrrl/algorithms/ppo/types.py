from typing import Any, NamedTuple

import numpy as np
import optax
from numpy.typing import NDArray

from brrrl.types import Params, PRNGKey


class PPOOutput(NamedTuple):
    action: NDArray
    log_prob: NDArray[np.float32]
    value: NDArray[np.float32]


class PPOTransition(NamedTuple):
    """One step. Stacked to ``(T, N, ...)`` as the buffered timescale."""

    observation: NDArray
    action: NDArray
    reward: NDArray[np.float32]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    log_prob: NDArray[np.float32]
    value: NDArray[np.float32]


class Bootstrap(NamedTuple):
    """Single-item timescale: every leaf ``(1, N, ...)``."""

    observation: NDArray


class PPOTrajectory(NamedTuple):
    transitions: PPOTransition
    bootstrap: Bootstrap


class PPOLearnerState(NamedTuple):
    params: Params
    opt_state: optax.OptState
    key: PRNGKey


def example(num_steps: int, num_lanes: int, obs_shape: tuple[int, ...]) -> dict[str, Any]:
    """Zero-filled timescales; shared verbatim by the accumulator, client and server."""
    raise NotImplementedError
