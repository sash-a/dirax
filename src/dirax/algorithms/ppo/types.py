from collections.abc import Callable
from typing import Any, NamedTuple, Self

import distrax
import gymnasium as gym
import jax
import numpy as np
import optax
import optree as tree

from dirax.types import HIDDEN_STATE, TRANSITIONS, Array, ArrayTree, Params, PRNGKey

type PPOApply = Callable[[Params, Any, ArrayTree], tuple[distrax.Distribution, jax.Array]]
"""What PPO's loss expects: (params, observations, per rollout) -> (policy distribution, value)"""


class PPOPolicyOutput(NamedTuple):
    action: Array
    log_prob: Array
    value: Array


class PPOPolicyInput(NamedTuple):
    observation: ArrayTree

    @classmethod
    def of(cls, obs: ArrayTree, term: Array, trunc: Array) -> Self:
        return cls(obs)


class RecPPOPolicyInput(NamedTuple):
    observation: ArrayTree
    dones: Array

    @classmethod
    def of(cls, obs: ArrayTree, term: Array, trunc: Array) -> Self:
        return cls(obs, term | trunc)


class RecPPOPolicyOutput(NamedTuple):
    action: Array
    log_prob: Array
    value: Array
    hstate: ArrayTree


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

    def per_rollout(self) -> ArrayTree:
        """A feed-forward run has no per-rollout context."""
        return ()


class RecPPOTrajectory(NamedTuple):
    transitions: PPOTransition
    hidden_state: ArrayTree

    def per_rollout(self) -> ArrayTree:
        """The carry the act behind `transitions[0]` consumed, with the width-one time axis dropped."""
        return jax.tree.map(lambda leaf: leaf[:, 0], self.hidden_state)


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
            all. `dirax.acting.env_stepper.prefix` composes it. The env axis leads either way, so the
            observation takes its first entry and nothing else.
        obs_space: One env's observation space, not the batched one.
        act_space: One agent's action space. `dirax.wrappers.agents.one_agent` of what the env
            declares.
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


def rec_example(
    rollout_length: int,
    hidden_size: int,
    prefix: tuple[int, ...],
    obs_space: gym.Space,
    act_space: gym.Space,
) -> dict[str, Any]:
    """Makes a recurrent example for echo.

    Args:
       rollout_length: Transitions per rollout. One slot more than this is filled, for the trailing
           bootstrap step.
       hidden_size: size of the flat hidden state. This function does not support non-flat array hidden states.
       prefix: The leading shape every field a stepper reports carries — ``(E, A)`` for an env with
           a team acting in it, and ``(E,)`` for one with a single agent, which has no agent axis at
           all. `dirax.acting.env_stepper.prefix` composes it. The env axis leads either way, so the
           observation takes its first entry and nothing else.
       obs_space: One env's observation space, not the batched one.
       act_space: One agent's action space. `dirax.wrappers.agents.one_agent` of what the env
           declares.
    """
    transition = example(rollout_length, prefix, obs_space, act_space)
    # The leading 1 is what makes this a single-item timescale to the accumulator, and what lands
    # in the time slot when `batch_major` rearranges it.
    hidden = {HIDDEN_STATE: np.empty((1, *prefix, hidden_size), np.float32)}
    return transition | hidden
