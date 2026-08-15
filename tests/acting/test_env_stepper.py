"""The construction check, which is the only thing a stepper has that can fail loudly.

ADR-0001 chose next-step autoreset because same-step returns an observation belonging to a
different episode than the flag beside it. A Roller carrying that observation forward would lay
a *reset* observation into a slot flagged terminated — wrong data that still trains, which is
the failure the check exists to turn into a refusal.
"""

from typing import cast

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.vector import AutoresetMode

from brrrl.acting.env_stepper import GymStepper, GymVectorStepper, applied

NUM_ENVS = 2


def cartpole() -> gym.Env:
    return gym.make("CartPole-v1")


def cartpoles() -> gym.vector.VectorEnv:
    """The factory a caller reaches for out of habit, which the stepper has to refuse."""
    return gym.make_vec("CartPole-v1", num_envs=NUM_ENVS)


def normalising_cartpole() -> gym.Env:
    """The composition a run scaling its rewards writes: statistics under the scaling, both in the
    factory, because only what builds an env knows what else is transforming it."""
    return gym.wrappers.NormalizeReward(gym.wrappers.RecordEpisodeStatistics(cartpole()))


def scaled_vector() -> GymVectorStepper:
    return GymVectorStepper(normalising_cartpole, NUM_ENVS)


def scaled_single() -> GymStepper:
    return GymStepper(normalising_cartpole())


def scaled_across_envs() -> GymVectorStepper:
    """The seam: one transform over all the envs at once, applied where the statistics are already
    under it. Nothing about the reward is the stepper's to know."""
    return GymVectorStepper(cartpole, NUM_ENVS, wrap=gym.wrappers.vector.NormalizeReward)


def stepped(stepper: GymVectorStepper) -> np.ndarray:
    """What one step pays, from a fresh reset."""
    stepper.reset()
    return stepper.step(np.zeros(NUM_ENVS, int))[1]


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


def test_an_env_that_already_carries_a_stepper_owned_wrapper_is_refused() -> None:
    """The autoreset is the stepper's, because the Trajectory layout rests on it (ADR-0001). A second
    one inserts an extra ignored step, and says nothing about it."""
    with pytest.raises(ValueError, match=gym.wrappers.Autoreset.__name__):
        GymStepper(gym.wrappers.Autoreset(gym.make("CartPole-v1")))


def test_the_vector_stepper_refuses_an_owned_wrapper_one_level_down() -> None:
    """The vector env owns the autoreset across all of its envs, so a per-env copy is the same
    double-wrap."""
    with pytest.raises(ValueError, match=gym.wrappers.Autoreset.__name__):
        GymVectorStepper(lambda: gym.wrappers.Autoreset(gym.make("CartPole-v1")), NUM_ENVS)


def test_a_reward_transform_composed_by_the_caller_reaches_the_step() -> None:
    """CartPole pays exactly one a step, so anything but one is the caller's wrapper and nothing else.

    Scaling is composed into the factory rather than asked of the stepper: what a reward means is
    the environment's business, and a stepper that owned it would own every other transform too.
    """
    plain, scaled = GymVectorStepper(cartpole, NUM_ENVS), GymVectorStepper(normalising_cartpole, NUM_ENVS)

    rewards = [stepped(stepper) for stepper in (plain, scaled)]

    assert np.all(rewards[0] == 1.0)
    assert not np.any(rewards[1] == 1.0)


def test_a_wrap_is_handed_the_envs_the_stepper_built() -> None:
    """Scaling across every env at once needs them all in one place, which only the stepper has. The
    seam takes them without the stepper learning what a reward is: what it hands over is already
    autoreset and already recorded, so neither of its own guarantees is on offer to break.
    """
    plain, scaled = GymVectorStepper(cartpole, NUM_ENVS), scaled_across_envs()

    rewards = [stepped(stepper) for stepper in (plain, scaled)]

    assert np.all(rewards[0] == 1.0)
    assert not np.any(rewards[1] == 1.0)


@pytest.mark.parametrize(
    "stepper", [scaled_vector, scaled_single, scaled_across_envs], ids=["vector", "single", "across envs"]
)
def test_an_episode_return_is_reported_in_the_environment_s_own_units(stepper) -> None:
    """The statistics are taken under the scaling. Above it, every run's returns would be divided by
    a spread that moves as the policy does, and no two runs could be compared at all.
    """
    stepping = stepper()
    stepping.reset()

    returns = [
        info["episode"]["r"][info["_episode"]]
        for _ in range(600)
        if "episode" in (info := stepping.step(np.zeros(stepping.num_envs, int))[4])
    ]

    assert returns, "no episode finished, so the assertion below would hold vacuously"
    assert all(finished == finished.astype(int) for finished in np.concatenate(returns))


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda env: GymStepper(env), id="single"),
        pytest.param(lambda env: GymVectorStepper(lambda: env, NUM_ENVS), id="vector"),
    ],
)
def test_scaling_a_reward_above_the_statistics_is_refused(build) -> None:
    """The one arrangement a stepper cannot fix for the caller: it applies statistics outermost, so
    an env arriving already scaled has to bring its own, underneath. Getting this wrong is silent —
    the returns still log, in units nothing downstream can convert back."""
    with pytest.raises(ValueError, match="scaled units"):
        build(gym.wrappers.NormalizeReward(cartpole()))


def test_statistics_a_caller_placed_are_not_applied_twice() -> None:
    """A second one reports the wrong episode, and says nothing about it."""
    stepper = GymVectorStepper(normalising_cartpole, NUM_ENVS)

    envs = cast(gym.vector.SyncVectorEnv, stepper.env).envs
    recording = [wrapper for wrapper in applied(envs[0]) if wrapper is gym.wrappers.RecordEpisodeStatistics]

    assert len(recording) == 1


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
