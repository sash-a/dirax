"""What one env has to answer for a Roller to read it and a vectorised one the same way.

Its autoreset discipline and its statistics are held to gymnasium's own vector env instead, in
``test_rollout.py`` — a hand-rolled reset deserves an oracle rather than our own assertions.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from brrrl.wrappers.action_mask import ACTION_MASK, OBSERVATION, ActionMask
from brrrl.wrappers.episodes import EPISODE_MASK, EPISODE_STATS
from brrrl.wrappers.looped import Looped

CARTPOLE_OBS = 4
TAXI_ACTIONS = 6


def cartpole() -> gym.Env:
    return gym.make("CartPole-v1")


def looped() -> Looped:
    return Looped([gym.wrappers.RecordEpisodeStatistics(cartpole())])


def test_every_field_leads_with_an_env_axis() -> None:
    """The one API a Roller steps, whatever built the environments."""
    env = looped()

    observation, _ = env.reset()
    stepped, reward, terminated, truncated, _ = env.step(np.zeros(1, int))

    assert observation.shape == stepped.shape == (1, CARTPOLE_OBS)
    assert reward.shape == terminated.shape == truncated.shape == (1,)


def test_the_spaces_say_so_too() -> None:
    """A caller sizing a buffer off the space gets the shape the steps actually arrive in."""
    env = looped()

    assert env.observation_space.shape == (1, CARTPOLE_OBS)
    assert env.action_space.shape == (1,)


def test_an_observation_is_a_fresh_array_each_step() -> None:
    """A Roller carries an observation across a step, so a buffer reused between steps would rewrite
    the slot it already recorded — with the *next* state, under the flags of this one."""
    env = looped()

    first, _ = env.reset()
    kept = first.copy()
    for _ in range(3):
        env.step(np.ones(1, int))

    assert np.array_equal(first, kept)


def test_the_action_is_unwrapped_on_the_way_in() -> None:
    """It arrives with the env axis every other field carries, and a `gym.Env` takes none."""
    env = looped()
    env.reset()

    left, _, _, _, _ = env.step(np.zeros(1, int))
    right, _, _, _, _ = env.step(np.ones(1, int))

    assert not np.array_equal(left, right)


def test_a_finished_episode_is_reported_in_the_vector_schema() -> None:
    """Full-width statistics under ``episode``, and the mask naming which envs they belong to. A
    Roller's episode reader knows that schema and nothing else, so one env has to speak it."""
    env = Looped([gym.wrappers.Autoreset(gym.wrappers.RecordEpisodeStatistics(cartpole()))])
    env.reset()

    infos = [info for _ in range(600) if EPISODE_STATS in (info := env.step(np.zeros(1, int))[4])]

    assert infos, "no episode finished, so the assertions below would hold vacuously"
    assert all(np.shape(info[EPISODE_STATS]["r"]) == (1,) for info in infos)
    assert all(info[EPISODE_MASK].tolist() == [True] for info in infos)


def test_an_info_without_statistics_passes_through_untouched() -> None:
    """Most steps finish no episode, and inventing a zero for those would report an episode that
    never happened."""
    env = looped()

    _, info = env.reset()

    assert EPISODE_STATS not in info and EPISODE_MASK not in info


def test_a_dict_observation_widens_leaf_by_leaf() -> None:
    """Which is what lets an action mask ride in the observation and need nothing else changed.

    Taxi rather than a stand-in, because this is the composition a masked run actually builds. It has
    one agent, so its mask carries no agent axis and the env axis is the only one in front of it.
    """
    env = Looped([ActionMask(gym.make("Taxi-v4"))])

    observation, _ = env.reset()

    assert observation[OBSERVATION].shape == (1,)
    assert observation[ACTION_MASK].shape == (1, TAXI_ACTIONS)


@pytest.mark.parametrize("field", ["reward", "terminated", "truncated"])
def test_a_looped_env_is_no_longer_an_ordinary_one(field: str) -> None:
    """It goes on outermost: a gymnasium wrapper above it would read an array where it expects a
    scalar reward or a bare flag, and most of them would keep going."""
    env = looped()
    env.reset()

    stepped: dict[str, Any] = dict(zip(["reward", "terminated", "truncated"], env.step(np.zeros(1, int))[1:4]))

    assert isinstance(stepped[field], np.ndarray)
