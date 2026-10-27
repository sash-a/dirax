from dataclasses import dataclass
from typing import Any, Protocol, final

import jax
from jax import tree

from dirax.types import PRNGKey


def leading_shape(batch: Any) -> tuple[int, int]:
    """The leading 2 dims of a batch, which a `Minibatcher` reads as (sequences, steps)."""
    B, T, *_ = tree.leaves(batch)[0].shape
    return B, T


class Minibatcher(Protocol):
    """How one update's batch is permuted and cut into the minibatches an epoch descends."""

    def validate(self, batch_size: int, rollout_length: int, num_minibatches: int) -> bool:
        """Whether ``num_minibatches`` divides the pool this cuts, which is what `isvalid` asks."""
        ...

    def cut[B, Q](self, key: PRNGKey, batch: B, per_rollout: Q, num_minibatches: int) -> tuple[B, Q]:
        """``batch`` and ``per_rollout`` permuted together and split along a new leading minibatch axis.

        Args:
            key: random key used for shuffling.
            batch: trajectory data of shape (B, T, ...).
            per_rollout: once per rollout data of shape (B, ...) e.g a hidden state.
            num_minibatches: the number of minibatch to split the batch and per rollout into.
        """
        ...


@final
@dataclass(frozen=True)
class Steps:
    """A flat pool of single steps, so a minibatch mixes along the time and batch dim.
    A cut comes out (num_minibatches, minibatch_size, ...).

    Decorrelates a minibatch the most, and destroys the time axis doing it, so it suits a
    feed-forward assembly only.
    """

    def validate(self, batch_size: int, rollout_length: int, num_minibatches: int) -> bool:
        return (batch_size * rollout_length) % num_minibatches == 0

    def cut[B, R](self, key: PRNGKey, batch: B, per_rollout: R, num_minibatches: int) -> tuple[B, R]:
        if tree.leaves(per_rollout):
            raise ValueError(
                "A flat pool of steps has no per-sequence axis for a per-rollout value to survive, so "
                "this cannot cut one. A recurrent run wants `Sequences`."
            )
        B, T = leading_shape(batch)
        flat = B * T
        size = flat // num_minibatches
        perm = jax.random.permutation(key, flat)
        cut = tree.map(
            lambda leaf: (
                leaf.reshape(flat, *leaf.shape[2:])  # flatten B, T
                .take(perm, axis=0)  # shuffle
                .reshape(num_minibatches, size, *leaf.shape[2:])  # batch into minibatches
            ),
            batch,
        )
        return cut, per_rollout


@final
@dataclass(frozen=True)
class Sequences:
    """A pool of whole sequences, permuted between sequences and never within one.
    A cut comes out ``(num_minibatches, T, minibatch_size, ...)``.
    """

    def validate(self, batch_size: int, rollout_length: int, num_minibatches: int) -> bool:
        return batch_size % num_minibatches == 0

    def cut[B, R](self, key: PRNGKey, batch: B, per_rollout: R, num_minibatches: int) -> tuple[B, R]:
        B, _ = leading_shape(batch)
        size = B // num_minibatches
        perm = jax.random.permutation(key, B)
        cut = tree.map(
            lambda leaf: (
                leaf.take(perm, axis=0)  # shuffle whole sequences
                .reshape(num_minibatches, size, *leaf.shape[1:])  # batch into minibatches
                .swapaxes(1, 2)
            ),  # time-major within a minibatch, which is what a scan takes
            batch,
        )
        # The same permutation, so a sequence keeps the context recorded alongside it. Nothing to
        # make time-major: there is one of these per rollout, not one per step.
        once = tree.map(
            lambda leaf: leaf.take(perm, axis=0).reshape(num_minibatches, size, *leaf.shape[1:]),
            per_rollout,
        )
        return cut, once
