from dataclasses import dataclass
from typing import Any, Protocol, final

import jax
from jax import tree

from brrrl.types import PRNGKey


def leading_shape(batch: Any) -> tuple[int, int]:
    """Returns the leading 2 dims of an array (usually T, B)"""
    T, B, *_ = tree.leaves(batch)[0].shape
    return T, B


class Minibatcher(Protocol):
    """How one update's batch is permuted and cut into the minibatches an epoch descends."""

    def validate(self, batch_size: int, rollout_length: int, num_minibatches: int) -> bool:
        """Whether ``num_minibatches`` divides the pool this cuts, which is what `isvalid` asks."""
        ...

    def cut[B](self, key: PRNGKey, batch: B, num_minibatches: int) -> B:
        """``batch`` permuted and split along a new leading minibatch axis."""
        ...


@final
@dataclass(frozen=True)
class Steps:
    """A flat pool of single steps, so a minibatch mixes along the time and batch dim.

    Decorrelates a minibatch the most, and destroys the time axis doing it, so it suits a
    feed-forward assembly only.
    """

    def validate(self, batch_size: int, rollout_length: int, num_minibatches: int) -> bool:
        return (batch_size * rollout_length) % num_minibatches == 0

    def cut[B](self, key: PRNGKey, batch: B, num_minibatches: int) -> B:
        T, B = leading_shape(batch)
        flat = T * B
        size = flat // num_minibatches
        perm = jax.random.permutation(key, flat)
        return tree.map(
            lambda leaf: (
                leaf.reshape(flat, *leaf.shape[2:])  # flatten T, B
                .take(perm, axis=0)  # shuffle
                .reshape(num_minibatches, size, *leaf.shape[2:])  # batch into minibatches
            ),
            batch,
        )


@final
@dataclass(frozen=True)
class Sequences:
    """A pool of whole sequences, permuted between sequences and never within one.

    Each minibatch comes out time-major, ``(T, sequences, ...)``
    """

    def validate(self, batch_size: int, rollout_length: int, num_minibatches: int) -> bool:
        return batch_size % num_minibatches == 0

    def cut[B](self, key: PRNGKey, batch: B, num_minibatches: int) -> B:
        T, _ = leading_shape(batch)
        size = T // num_minibatches
        perm = jax.random.permutation(key, T)
        return tree.map(
            lambda leaf: (
                leaf.take(perm, axis=0)  # shuffle whole sequences
                .reshape(num_minibatches, size, *leaf.shape[1:])  # batch into minibatches
                .swapaxes(1, 2)
            ),  # time-major within a minibatch, which is what a scan takes
            batch,
        )
