"""How a batch is permuted and cut, which is the one thing recurrence changes about an update.

Nothing here is PPO's. A Minibatcher reads the ``(sequences, moments)`` a batch leads with off any
leaf, so a batch double of two arrays exercises it exactly as an algorithm's own does — and a test
that had to build a `PPOBatch` would be evidence the layering had slipped back.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dirax.learning.minibatcher import Minibatcher, Sequences, Steps

SEQUENCES, MOMENTS, MINIBATCHES = 4, 6, 2
TRAILING = 3


class Batch(NamedTuple):
    """A batch double: one leaf with a trailing axis of its own and one without."""

    marks: jax.Array
    widened: jax.Array


def stamped() -> Batch:
    """Every entry reads ``sequence * 100 + moment``, so a cut can be read back."""
    marks = jnp.array([[row * 100 + at for at in range(MOMENTS)] for row in range(SEQUENCES)], jnp.float32)
    return Batch(marks, jnp.broadcast_to(marks[..., None], (SEQUENCES, MOMENTS, TRAILING)))


@pytest.mark.parametrize("minibatcher", [Steps(), Sequences()], ids=["steps", "sequences"])
def test_a_cut_loses_no_entry_and_duplicates_none(minibatcher: Minibatcher) -> None:
    cut, _ = minibatcher.cut(jax.random.key(0), stamped(), (), MINIBATCHES)

    assert sorted(np.asarray(cut.marks).ravel().tolist()) == sorted(np.asarray(stamped().marks).ravel().tolist())


@pytest.mark.parametrize("minibatcher", [Steps(), Sequences()], ids=["steps", "sequences"])
def test_a_cut_carries_whatever_trailing_axes_a_leaf_had(minibatcher: Minibatcher) -> None:
    """A leaf is cut by its leading axes alone, so a continuous action's axis or a mask's survives."""
    cut, _ = minibatcher.cut(jax.random.key(0), stamped(), (), MINIBATCHES)

    assert cut.widened.shape == (*cut.marks.shape, TRAILING)


def test_steps_pools_every_step_and_drops_the_time_axis() -> None:
    cut, _ = Steps().cut(jax.random.key(0), stamped(), (), MINIBATCHES)

    assert cut.marks.shape == (MINIBATCHES, SEQUENCES * MOMENTS // MINIBATCHES)


def test_sequences_cuts_time_major_so_a_scan_can_take_a_minibatch_whole() -> None:
    """`ResetRecurrence` scans the leading axis, so a minibatch of sequences has to lead with time."""
    cut, _ = Sequences().cut(jax.random.key(0), stamped(), (), MINIBATCHES)

    assert cut.marks.shape == (MINIBATCHES, MOMENTS, SEQUENCES // MINIBATCHES)


def test_sequences_keeps_every_sequence_whole_and_in_order() -> None:
    """A scan over a minibatch has to be a scan over moments one environment really lived through."""
    cut, _ = Sequences().cut(jax.random.key(0), stamped(), (), MINIBATCHES)

    for minibatch in range(MINIBATCHES):
        for column in np.asarray(cut.marks[minibatch]).T:
            assert len({int(mark // 100) for mark in column}) == 1, "a column mixes sequences"
            assert [int(mark % 100) for mark in column] == list(range(MOMENTS)), "a column is out of order"


def test_steps_mixes_moments_across_sequences() -> None:
    """The property it exists for, and the one that makes it useless to a recurrence."""
    cut, _ = Steps().cut(jax.random.key(0), stamped(), (), MINIBATCHES)

    assert any(len({int(mark // 100) for mark in np.asarray(row)}) > 1 for row in cut.marks)


def test_each_minibatcher_validates_the_pool_it_actually_cuts() -> None:
    """A count that divides a flat pool of steps need not divide a pool of sequences, so `isvalid`
    has to ask the strategy rather than assume one."""
    assert Steps().validate(SEQUENCES, MOMENTS, SEQUENCES * MOMENTS)
    assert not Sequences().validate(SEQUENCES, MOMENTS, SEQUENCES * MOMENTS)
    assert Sequences().validate(SEQUENCES, MOMENTS, SEQUENCES)


@pytest.mark.parametrize("minibatcher", [Steps(), Sequences()], ids=["steps", "sequences"])
def test_a_minibatcher_is_hashable_so_a_rule_holding_one_stays_a_static_jit_argument(minibatcher: Minibatcher) -> None:
    assert hash(minibatcher) == hash(type(minibatcher)())


def once_per_rollout() -> jax.Array:
    """One value recorded for the rollout, stamped with its sequence's index."""
    return jnp.arange(SEQUENCES, dtype=jnp.float32)[:, None] * 100.0


def test_sequences_keeps_each_sequences_per_rollout_value_with_it() -> None:
    """The permutation has to be shared. A sequence cut away from the context recorded with it is a
    loss recomputing that sequence from another one's memory, and nothing downstream would notice.
    """
    cut, per_rollout = Sequences().cut(jax.random.key(0), stamped(), once_per_rollout(), MINIBATCHES)

    for minibatch in range(MINIBATCHES):
        moments = np.asarray(cut.marks[minibatch])  # (moments, sequences)
        recorded = np.asarray(per_rollout[minibatch])  # (sequences, 1)
        for column, alongside in zip(moments.T, recorded, strict=True):
            assert int(column[0] // 100) == int(alongside[0] // 100), "a sequence lost its context"


def test_a_per_rollout_value_is_cut_without_a_time_axis() -> None:
    """There is one per sequence, not one per moment, so it is cut at a different rate."""
    cut, per_rollout = Sequences().cut(jax.random.key(0), stamped(), once_per_rollout(), MINIBATCHES)

    assert cut.marks.shape == (MINIBATCHES, MOMENTS, SEQUENCES // MINIBATCHES)
    assert per_rollout.shape == (MINIBATCHES, SEQUENCES // MINIBATCHES, 1)


def test_an_empty_per_rollout_value_survives_either_cut_as_nothing() -> None:
    """What a feed-forward run passes, so neither cutter needs to know whether it has one."""
    for minibatcher in (Steps(), Sequences()):
        _, per_rollout = minibatcher.cut(jax.random.key(0), stamped(), (), MINIBATCHES)
        assert per_rollout == ()


def test_steps_refuses_a_per_rollout_value_it_cannot_cut() -> None:
    """A flat pool of steps has no sequence axis to attach one to, so pairing them is a config
    error rather than something to silently drop."""
    with pytest.raises(ValueError, match="per-sequence axis"):
        Steps().cut(jax.random.key(0), stamped(), once_per_rollout(), MINIBATCHES)
