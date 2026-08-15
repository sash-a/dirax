"""The construction check, which is the only thing a stepper has that can fail loudly.

ADR-0001 chose next-step autoreset because same-step returns an observation belonging to a
different episode than the flag beside it. A Roller carrying that observation forward would lay
a *reset* observation into a slot flagged terminated — wrong data that still trains, which is
the failure the check exists to turn into a refusal.
"""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.vector import AutoresetMode

from brrrl.acting.env_stepper import GymStepper, GymVectorStepper

NUM_ENVS = 2


def cartpole() -> gym.Env:
    return gym.make("CartPole-v1")


def cartpoles() -> gym.vector.VectorEnv:
    """The factory a caller reaches for out of habit, which the stepper has to refuse."""
    return gym.make_vec("CartPole-v1", num_envs=NUM_ENVS)


def test_a_factory_is_vectorised_to_the_count_it_was_given() -> None:
    """``num_envs`` is the second factor in a Roller's transition count, so only the stepper has it."""
    stepper = GymVectorStepper(cartpole, NUM_ENVS)

    assert stepper.num_envs == NUM_ENVS


def test_the_stepper_picks_next_step_autoreset_rather_than_checking_for_it() -> None:
    """The mode a caller used to be able to get wrong is now the stepper's own, so the refusal that
    used to guard it has become an invariant instead."""
    stepper = GymVectorStepper(cartpole, NUM_ENVS)

    assert stepper.env.metadata["autoreset_mode"] is AutoresetMode.NEXT_STEP


def test_an_already_vectorised_env_is_refused_at_construction() -> None:
    """It is handed what to vectorise. Something already vectorised carries an autoreset mode this
    class no longer inspects, which is exactly the assumption ADR-0001 refuses to rest on."""
    with pytest.raises(TypeError, match="already vectorised"):
        GymVectorStepper(cartpoles, NUM_ENVS)  # ty: ignore[invalid-argument-type]


def test_an_env_with_no_environments_is_refused() -> None:
    with pytest.raises(ValueError, match="num_envs"):
        GymVectorStepper(cartpole, 0)


def test_a_single_environment_is_an_ordinary_vectorised_one() -> None:
    """Debugging one env exercises the same code path as a hundred, and the vector statistics
    schema needs no hand-synthesising."""
    stepper = GymVectorStepper(cartpole, 1)

    observation, _ = stepper.reset()

    assert stepper.num_envs == 1
    assert observation.shape[0] == 1


@pytest.mark.parametrize("wrapper", [gym.wrappers.Autoreset, gym.wrappers.RecordEpisodeStatistics])
def test_an_env_that_already_carries_a_stepper_owned_wrapper_is_refused(wrapper: type[gym.Wrapper]) -> None:
    """A stepper applies both of these itself. A second autoreset inserts an extra ignored step and
    a second statistics wrapper reports the wrong episode, and neither of them says so."""
    with pytest.raises(ValueError, match=wrapper.__name__):
        GymStepper(wrapper(gym.make("CartPole-v1")))


@pytest.mark.parametrize("wrapper", [gym.wrappers.Autoreset, gym.wrappers.RecordEpisodeStatistics])
def test_the_vector_stepper_refuses_an_owned_wrapper_one_level_down(wrapper: type[gym.Wrapper]) -> None:
    """The vector env owns both across all of its envs, so a per-env copy is the same double-wrap."""
    with pytest.raises(ValueError, match=wrapper.__name__):
        GymVectorStepper(lambda: wrapper(gym.make("CartPole-v1")), NUM_ENVS)


def test_a_single_env_stepper_puts_an_env_axis_on_every_field() -> None:
    """The one API a Roller steps: whatever built the environments, every field leads with envs.

    Its autoreset discipline and its statistics are held to gymnasium's own vector env instead, in
    ``test_rollout.py`` — a hand-rolled reset deserves an oracle rather than our own assertions.
    """
    stepper = GymStepper(gym.make("CartPole-v1"))

    observation, _ = stepper.reset()
    stepped, reward, terminated, truncated, _ = stepper.step(np.zeros(1, int))

    assert stepper.num_envs == 1
    assert observation.shape == stepped.shape == (1, 4)
    assert reward.shape == terminated.shape == truncated.shape == (1,)
