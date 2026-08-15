"""An observation carrying the legal actions an env reported beside it.

The masking itself is the Head's, in ``networks/test_head.py``. What is here is only that the mask
an env published reaches the observation, unchanged and beside the state it describes.
"""

from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.vector import AutoresetMode

from brrrl.wrappers.action_mask import ACTION_MASK, OBSERVATION, ActionMask

CARTPOLE_OBS = 4
ACTIONS = 2
NUM_ENVS = 3
FORBIDDEN = 1
LEGAL = np.array([action != FORBIDDEN for action in range(ACTIONS)], np.int8)
ELSEWHERE = "legal_moves"


def cartpole() -> gym.Env:
    """Reports no mask, which is what most envs do."""
    return gym.make("CartPole-v1")


class Publishing(gym.Wrapper):
    """An env reporting its legal set the way gymnasium's own masked envs do, under ``key``.

    CartPole masks nothing, and every env that does is either a dependency or a hundred lines of
    grid world. This is the smallest thing that behaves like one.
    """

    def __init__(self, env: gym.Env, key: str = ACTION_MASK) -> None:
        super().__init__(env)
        self.key = key

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        return observation, info | {self.key: LEGAL}

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info | {self.key: LEGAL}


def masked(key: str = ACTION_MASK) -> gym.Env:
    return ActionMask(Publishing(cartpole(), key))


def test_the_observation_space_carries_both_halves() -> None:
    """A Trajectory is sized off this space, so a mask that is not in it gets no slot to travel in."""
    space = masked().observation_space

    assert isinstance(space, gym.spaces.Dict)
    assert space[OBSERVATION].shape == (CARTPOLE_OBS,)
    assert space[ACTION_MASK].shape == (ACTIONS,)


@pytest.mark.parametrize("act", [lambda env: env.reset(), lambda env: env.step(0)], ids=["reset", "step"])
def test_the_mask_an_env_reported_is_the_one_carried(act) -> None:
    """Read out of the info beside a step rather than off the env, which is where an env that has a
    mask puts it — and re-deriving it from the env's own state means reaching past every wrapper."""
    env = masked()
    env.reset()

    observation = act(env)[0]

    assert set(observation) == {OBSERVATION, ACTION_MASK}
    assert observation[ACTION_MASK].tolist() == LEGAL.tolist()


def test_an_env_reporting_its_mask_elsewhere_is_read_by_a_function_that_knows_where() -> None:
    """One convention exists and not every env follows it, so where to look is the caller's to say."""
    env = ActionMask(Publishing(cartpole(), ELSEWHERE), lambda info: info[ELSEWHERE])

    observation, _ = env.reset()

    assert observation[ACTION_MASK].tolist() == LEGAL.tolist()


def test_an_env_reporting_no_mask_at_all_says_so() -> None:
    """Wrapping an env that masks nothing is the mistake, so it fails at the first step rather than
    training against a mask quietly invented for it."""
    with pytest.raises(KeyError, match=ACTION_MASK):
        ActionMask(cartpole()).reset()


def test_the_mask_matches_the_space_it_is_declared_in() -> None:
    """A Trajectory slot is allocated from the space, so a mask of another dtype is cast on every
    write — silently, and on the hot path."""
    env = masked()
    space = cast(gym.spaces.Dict, env.observation_space)

    observation, _ = env.reset()

    assert observation[ACTION_MASK].dtype == space[ACTION_MASK].dtype
    assert space.contains(observation)


def test_the_underlying_observation_is_passed_through_unchanged() -> None:
    plain, wrapped = cartpole(), masked()

    expected, _ = plain.reset(seed=0)
    observation, _ = wrapped.reset(seed=0)

    assert np.array_equal(observation[OBSERVATION], expected)


def test_the_info_is_passed_through_unchanged() -> None:
    """A mask reaching the observation does not take it out of the info: a Roller reads episode
    statistics from the same dict, and this wrapper has no business editing it."""
    env = masked()

    _, info = env.reset()

    assert info[ACTION_MASK].tolist() == LEGAL.tolist()


def test_a_continuous_action_space_is_refused() -> None:
    """There is nothing to mask off a real interval, and an all-true mask over one is a leaf on every
    hop a Trajectory makes for a value nothing ever reads."""
    with pytest.raises(TypeError, match="mask off"):
        ActionMask(gym.make("Pendulum-v1"))


def test_masked_envs_vectorise_into_one_batched_observation() -> None:
    """The wrapper is per env and the mask still arrives with the env axis every other field has."""
    envs = gym.vector.SyncVectorEnv([masked] * NUM_ENVS, autoreset_mode=AutoresetMode.NEXT_STEP)

    observation, _ = envs.reset()

    assert observation[OBSERVATION].shape == (NUM_ENVS, CARTPOLE_OBS)
    assert observation[ACTION_MASK].shape == (NUM_ENVS, ACTIONS)
    assert np.array_equal(observation[ACTION_MASK][:, FORBIDDEN], np.zeros(NUM_ENVS))
