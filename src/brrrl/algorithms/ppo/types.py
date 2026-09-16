from collections.abc import Callable
from typing import Any, NamedTuple, Self

import distrax
import gymnasium as gym
import jax
import numpy as np
import optax
import optree as tree

from brrrl.types import TRANSITIONS, Array, ArrayTree, Params, PRNGKey

type PPOApply = Callable[[Params, ArrayTree], tuple[distrax.Distribution, jax.Array]]


class PPOPolicyOutput(NamedTuple):
    action: Array
    log_prob: Array
    value: Array


class PPOTransition(NamedTuple):
    observation: ArrayTree
    terminated: Array
    truncated: Array
    reward: Array
    action: Array
    value: Array
    log_prob: Array

    @classmethod
    def record(cls, obs: ArrayTree, term: Array, trunc: Array, reward: Array, policy_output: PPOPolicyOutput) -> Self:
        return cls(obs, term, trunc, reward, policy_output.action, policy_output.value, policy_output.log_prob)


class PPOTrajectory(NamedTuple):
    transitions: PPOTransition


class PPOBatch(NamedTuple):
    transitions: PPOTransition
    advantages: Array
    value_targets: Array


class PPOUpdateState(NamedTuple):
    params: Params
    opt_state: optax.OptState
    key: PRNGKey


def example(rollout_length: int, prefix: tuple[int, ...], obs_space: gym.Space, act_space: gym.Space) -> dict[str, Any]:
    """The buffers one rollout fills. ``act_space`` is one agent's, because that is what a Head emits.

    Args:
        rollout_length: Transitions per rollout. One slot more than this is filled, for the trailing
            bootstrap step.
        prefix: The leading shape every field a stepper reports carries — ``(E, A)`` for an env with
            a team acting in it, and ``(E,)`` for one with a single agent, which has no agent axis at
            all. `brrrl.acting.env_stepper.prefix` composes it. The env axis leads either way, so the
            observation takes its first entry and nothing else.
        obs_space: One env's observation space, not the batched one.
        act_space: One agent's action space. `brrrl.wrappers.agents.one_agent` of what the env
            declares.

    The observation is the one field that does not take the whole prefix. It keeps whatever shape
    its env gave it, so an env whose agents each observe for themselves already carries the agent
    axis there and one whose agents share a view does not — nothing in the stepper chain touches an
    observation, and unpacking it is the algorithm's job (ADR-0008).
    """
    traj_len = rollout_length + 1  # because there is a final bootstrap step
    num_envs = prefix[0]
    obs = tree.tree_map(lambda x: np.zeros((traj_len, num_envs, *np.shape(x)), np.asarray(x).dtype), obs_space.sample())
    # An action slot is the action space's own shape and dtype, so a continuous one keeps its axis.
    action = np.asarray(act_space.sample())

    # Everything here leads with the rollout axis and whatever the stepper reports beneath it.
    width = (traj_len, *prefix)
    trajectory = PPOTransition(
        observation=obs,  # type: ignore
        terminated=np.empty(width, dtype=np.bool_),
        truncated=np.empty(width, dtype=np.bool_),
        action=np.empty((*width, *action.shape), dtype=action.dtype),
        value=np.empty(width, dtype=np.float32),
        reward=np.empty(width, dtype=np.float32),
        log_prob=np.empty(width, dtype=np.float32),
    )

    return {TRANSITIONS: trajectory}
