from typing import Any, NamedTuple

import numpy as np
import optax
from numpy.typing import NDArray

from brrrl.types import Array, ArrayTree, Params, PRNGKey


class PPOOutput(NamedTuple):
    action: Array
    log_prob: Array
    value: Array


class PPOTransition(NamedTuple):
    """One slot, in the order it aligns. Stacked to ``(T+1, E, ...)`` as the buffered timescale."""

    observation: ArrayTree
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    reward: NDArray[np.floating]
    action: Array
    value: Array
    log_prob: Array


class PPOTrajectory(NamedTuple):
    transitions: PPOTransition
    version: NDArray[np.int32]


class PPOLearnerState(NamedTuple):
    params: Params
    opt_state: optax.OptState
    key: PRNGKey


def example(num_steps: int, num_envs: int, obs_shape: tuple[int, ...]) -> dict[str, Any]:
    """Zero-filled timescales; shared verbatim by the accumulator, client and server."""
    raise NotImplementedError
