"""The estimator's contract, asserted against rlax as an oracle.

rlax expresses an episode boundary as one ``discount_t``, while we keep terminated and truncated
apart and carry a Final obs slot (ADR-0001, ADR-0012). The translation between the two conventions
is written out in ``rlax_advantages`` below. Getting it wrong does not crash — it trains, just
worse — which is why rlax stays a dev dependency and this file exists.

The semantic tests do not go through rlax at all: they assert what a boundary *means* by changing
data on one side of it and watching what moves.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import rlax

from brrrl.algorithms.estimators import generalised_advantages

GAMMA, GAE_LAMBDA = 0.99, 0.95
SLOTS, ENVS = 9, 3
TERMINATION, TRUNCATION = 6, 3


def flags(terminated: tuple[int, ...] = (), truncated: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    """Boolean flags of ``SLOTS`` with the named slots set."""
    made = [np.zeros(SLOTS, bool), np.zeros(SLOTS, bool)]
    for array, slots in zip(made, (terminated, truncated), strict=True):
        array[list(slots)] = True
    return made[0], made[1]


@pytest.fixture
def rollout() -> tuple[np.ndarray, np.ndarray]:
    """Rewards and values distinct per slot, so any misalignment shows up as a wrong number."""
    generator = np.random.default_rng(0)
    return generator.normal(size=SLOTS).astype(np.float32), generator.normal(size=SLOTS).astype(np.float32)


def rlax_advantages(reward, value, terminated, truncated, gamma=GAMMA, gae_lambda=GAE_LAMBDA) -> np.ndarray:
    """The oracle, under the one translation that reconciles the two conventions.

    Both flags are read off the slot the step lands in. ``discount_t`` goes to zero into a
    termination, so nothing bootstraps past it. ``lambda_`` goes to zero into a truncation, so the
    next episode's trace cannot walk backwards across the seam while the Final obs still bootstraps.
    ``sample_mask`` is deliberately not used: a masked step passes the accumulator through
    unchanged rather than resetting it, so it cannot carry a boundary.
    """
    return np.asarray(
        rlax.truncated_generalized_advantage_estimation(
            jnp.asarray(reward[:-1]),
            jnp.asarray(gamma * ~terminated[1:], jnp.float32),
            jnp.asarray(gae_lambda * ~truncated[1:], jnp.float32),
            jnp.asarray(value),
        )
    )


@pytest.mark.parametrize(
    ("terminated", "truncated"),
    [
        pytest.param((), (), id="unbroken"),
        pytest.param((TERMINATION,), (), id="termination"),
        pytest.param((), (TRUNCATION,), id="truncation"),
        pytest.param((TERMINATION,), (TRUNCATION,), id="both"),
        pytest.param((1, 5), (3, 7), id="many"),
        pytest.param((SLOTS - 1,), (), id="termination in the trailing slot"),
    ],
)
def test_matches_rlax_under_the_documented_translation(rollout, terminated, truncated) -> None:
    reward, value = rollout
    terminated, truncated = flags(terminated, truncated)

    advantages = generalised_advantages(reward, value, terminated, truncated, GAMMA, GAE_LAMBDA)

    assert np.allclose(advantages, rlax_advantages(reward, value, terminated, truncated), atol=1e-5)


def test_a_termination_severs_everything_after_it(rollout) -> None:
    """No reward or value from the next episode may reach an advantage in this one."""
    reward, value = rollout
    terminated, truncated = flags(terminated=(TERMINATION,))
    before = generalised_advantages(reward, value, terminated, truncated, GAMMA, GAE_LAMBDA)

    later_reward, later_value = reward.copy(), value.copy()
    later_reward[TERMINATION:] += 100.0
    later_value[TERMINATION:] += 100.0
    after = generalised_advantages(later_reward, later_value, terminated, truncated, GAMMA, GAE_LAMBDA)

    assert np.allclose(before[:TERMINATION], after[:TERMINATION], atol=1e-5)


def test_a_termination_reads_nothing_from_its_final_obs(rollout) -> None:
    """A termination bootstraps off nothing, so the value of the state it ended in is inert."""
    reward, value = rollout
    terminated, truncated = flags(terminated=(TERMINATION,))
    moved = value.copy()
    moved[TERMINATION] += 100.0

    before = generalised_advantages(reward, value, terminated, truncated, GAMMA, GAE_LAMBDA)
    after = generalised_advantages(reward, moved, terminated, truncated, GAMMA, GAE_LAMBDA)

    assert np.allclose(before[:TERMINATION], after[:TERMINATION], atol=1e-5)


def test_a_truncation_bootstraps_off_its_final_obs(rollout) -> None:
    """The one thing a truncation does read: its Final obs value, which a termination ignores."""
    reward, value = rollout
    terminated, truncated = flags(truncated=(TRUNCATION,))
    moved = value.copy()
    moved[TRUNCATION] += 100.0

    before = generalised_advantages(reward, value, terminated, truncated, GAMMA, GAE_LAMBDA)
    after = generalised_advantages(reward, moved, terminated, truncated, GAMMA, GAE_LAMBDA)

    assert not np.allclose(before[TRUNCATION - 1], after[TRUNCATION - 1], atol=1e-5)


def test_a_truncation_severs_the_trace_behind_it(rollout) -> None:
    """It bootstraps, but the next episode's accumulation still may not walk back across the seam."""
    reward, value = rollout
    terminated, truncated = flags(truncated=(TRUNCATION,))
    later_reward = reward.copy()
    later_reward[TRUNCATION + 1 :] += 100.0

    before = generalised_advantages(reward, value, terminated, truncated, GAMMA, GAE_LAMBDA)
    after = generalised_advantages(later_reward, value, terminated, truncated, GAMMA, GAE_LAMBDA)

    assert np.allclose(before[:TRUNCATION], after[:TRUNCATION], atol=1e-5)


def test_undiscounted_advantage_is_the_full_return_less_the_value(rollout) -> None:
    """With gamma and lambda at one and no boundary, GAE collapses to Monte Carlo — an analytic check."""
    reward, value = rollout
    terminated, truncated = flags()

    advantages = generalised_advantages(reward, value, terminated, truncated, gamma=1.0, gae_lambda=1.0)

    assert np.isclose(advantages[0], reward[:-1].sum() + value[-1] - value[0], atol=1e-4)


def test_the_slot_axis_is_a_parameter_rather_than_a_vmap(rollout) -> None:
    """A batch-major caller passes ``axis=1``; nothing needs wrapping."""
    reward, value = rollout
    terminated, truncated = flags(terminated=(TERMINATION,), truncated=(TRUNCATION,))
    rows = lambda leaf: np.stack([leaf] * ENVS, axis=0)

    time_major = generalised_advantages(reward, value, terminated, truncated, GAMMA, GAE_LAMBDA)
    batch_major = generalised_advantages(
        rows(reward), rows(value), rows(terminated), rows(truncated), GAMMA, GAE_LAMBDA, axis=1
    )

    assert batch_major.shape == (ENVS, SLOTS - 1)
    assert np.allclose(batch_major[0], time_major, atol=1e-6)


def test_trailing_axes_map_implicitly(rollout) -> None:
    """Slots down axis 0 and anything else beside it, the way ``jnp.sum`` broadcasts."""
    reward, value = rollout
    terminated, truncated = flags(terminated=(TERMINATION,))
    columns = lambda leaf: np.stack([leaf] * ENVS, axis=1)

    advantages = generalised_advantages(
        columns(reward), columns(value), columns(terminated), columns(truncated), GAMMA, GAE_LAMBDA
    )

    assert advantages.shape == (SLOTS - 1, ENVS)
    assert np.allclose(
        advantages[:, 0], generalised_advantages(reward, value, terminated, truncated, GAMMA, GAE_LAMBDA)
    )


def test_the_trailing_slot_is_consumed_rather_than_returned(rollout) -> None:
    """T+1 slots in, T advantages out: the trailing slot is only ever a bootstrap."""
    reward, value = rollout
    terminated, truncated = flags()

    assert generalised_advantages(reward, value, terminated, truncated, GAMMA, GAE_LAMBDA).shape == (SLOTS - 1,)
