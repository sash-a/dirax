from dataclasses import dataclass
from typing import Protocol, final

import jax
import jax.numpy as jnp
from jax import tree

from brrrl.algorithms.ppo.types import PPOBatch
from brrrl.types import PRNGKey


class Minibatcher(Protocol):
    """How one update's batch is permuted and cut into the minibatches an epoch descends.

    The one thing recurrence changes about the update: a scanned network needs a minibatch that is
    still a set of sequences, while a feed-forward one is free to mix moments from all of them.
    """

    def validate(self, batch_size: int, rollout_length: int, num_minibatches: int) -> int:
        """How many items there are to cut, which is what ``num_minibatches`` has to divide."""
        ...

    def cut(self, key: PRNGKey, batch: PPOBatch, num_minibatches: int) -> PPOBatch:
        """``batch`` permuted and split along a new leading minibatch axis."""
        ...


@final
@dataclass(frozen=True)
class Steps:
    """A flat pool of single steps, so a minibatch mixes moments from every sequence.

    Decorrelates a minibatch the most, and destroys the time axis doing it, so it suits a
    feed-forward assembly only.
    """

    def validate(self, batch_size: int, rollout_length: int, num_minibatches: int) -> int:
        return (batch_size * rollout_length) % num_minibatches == 0

    def cut(self, key: PRNGKey, batch: PPOBatch, num_minibatches: int) -> PPOBatch:
        flat = batch.advantages.shape[0] * batch.advantages.shape[1]
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

    Each minibatch comes out time-major, ``(T, sequences, ...)``, which is the layout
    `ResetRecurrence` scans — so a carry threaded through one runs the moments an environment
    actually lived through, in the order it lived them.
    """

    def validate(self, batch_size: int, rollout_length: int, num_minibatches: int) -> int:
        return batch_size % num_minibatches == 0

    def cut(self, key: PRNGKey, batch: PPOBatch, num_minibatches: int) -> PPOBatch:
        sequences = batch.advantages.shape[0]
        size = sequences // num_minibatches
        perm = jax.random.permutation(key, sequences)
        return tree.map(
            lambda leaf: jnp.swapaxes(
                leaf.take(perm, axis=0).reshape(  # shuffle whole sequences
                    num_minibatches, size, *leaf.shape[1:]
                ),  # batch into minibatches
                1,
                2,  # time-major within a minibatch, which is what a scan takes
            ),
            batch,
        )
