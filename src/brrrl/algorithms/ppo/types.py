from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np
import optax
import optree as tree
from numpy.typing import NDArray

from brrrl.types import TRANSITIONS, VERSION, Array, ArrayTree, Params, PRNGKey


class PPOPolicyOutput(NamedTuple):
    action: Array
    log_prob: Array
    value: Array


class PPOTransition(NamedTuple):
    observation: ArrayTree
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    reward: NDArray[np.floating]
    action: Array
    value: Array
    log_prob: Array


class PPOTrajectory(NamedTuple):
    transitions: PPOTransition
    version: NDArray[np.float32]


class PPOBatch(NamedTuple):
    transitions: PPOTransition
    advantages: Array
    value_targets: Array


class PPOUpdateState(NamedTuple):
    params: Params
    opt_state: optax.OptState
    key: PRNGKey


def example(rollout_length: int, num_envs: int, obs_space: gym.Space, act_space: gym.Space) -> dict[str, Any]:
    traj_len = rollout_length + 1  # because there is a final bootstrap step
    obs = tree.tree_map(lambda x: np.zeros((traj_len, num_envs, *np.shape(x)), np.asarray(x).dtype), obs_space.sample())
    # An action slot is the action space's own shape and dtype, so a continuous one keeps its axis.
    action = np.asarray(act_space.sample())

    # Everything here has a rollout dim
    trajectory = PPOTransition(
        observation=obs,  # type: ignore
        terminated=np.empty((traj_len, num_envs), dtype=np.bool_),
        truncated=np.empty((traj_len, num_envs), dtype=np.bool_),
        action=np.empty((traj_len, num_envs, *action.shape), dtype=action.dtype),
        value=np.empty((traj_len, num_envs), dtype=np.float32),
        reward=np.empty((traj_len, num_envs), dtype=np.float32),
        log_prob=np.empty((traj_len, num_envs), dtype=np.float32),
    )

    param_generation = np.empty((), dtype=np.float32)

    return {TRANSITIONS: trajectory, VERSION: param_generation}
