"""The construction check, which is the only thing a stepper has that can fail loudly.

ADR-0001 chose next-step autoreset because same-step returns an observation belonging to a
different episode than the flag beside it. A Roller carrying that observation forward would lay
a *reset* observation into a slot flagged terminated — wrong data that still trains, which is
the failure the check exists to turn into a refusal.
"""

import gymnasium as gym
import pytest
from gymnasium.vector import AutoresetMode, VectorEnv

from brrrl.acting.env_stepper import GymVectorStepper

NUM_ENVS = 2


def cartpoles(**kwargs: object) -> VectorEnv:
    return gym.make_vec("CartPole-v1", num_envs=NUM_ENVS, vectorization_mode="sync", vector_kwargs=kwargs)


def test_a_next_step_environment_is_accepted_and_reports_its_env_count() -> None:
    """``num_envs`` is the second factor in a Roller's transition count, so only the stepper has it."""
    stepper = GymVectorStepper(cartpoles())

    assert stepper.num_envs == NUM_ENVS


def test_a_same_step_environment_is_refused_at_construction() -> None:
    """Not at the first step, and not by training slightly worse: at the moment it is built."""
    with pytest.raises(ValueError, match="SAME_STEP"):
        GymVectorStepper(cartpoles(autoreset_mode=AutoresetMode.SAME_STEP))


def test_an_environment_that_declares_no_autoreset_mode_is_refused_too() -> None:
    """Gymnasium's own statistics wrapper *assumes* next-step here and warns. A layout that is
    wrong in a way that still trains cannot rest on an assumption, so an undeclared mode is
    refused rather than hoped about."""
    with pytest.raises(ValueError, match="autoreset"):
        GymVectorStepper(gym.vector.VectorEnv())


def test_a_single_environment_is_an_ordinary_vectorised_one() -> None:
    """Debugging one env exercises the same code path as a hundred, and the vector statistics
    schema needs no hand-synthesising."""
    stepper = GymVectorStepper(gym.make_vec("CartPole-v1", num_envs=1, vectorization_mode="sync"))

    observation, _ = stepper.reset()

    assert stepper.num_envs == 1
    assert observation.shape[0] == 1
