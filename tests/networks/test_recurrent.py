"""The learn-time shape only.

Nothing here decides how hidden state crosses the act boundary — whether a Policy carries it
onto the wire or an inference server keys it per Lane is a separate, open question.
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from brrrl.networks.recurrent import ResetRecurrence

NUM_STEPS = 5
NUM_LANES = 4
FEATURES = 6
HIDDEN = 8


def inputs(num_steps: int = NUM_STEPS) -> jax.Array:
    return jax.random.normal(jax.random.key(0), (num_steps, NUM_LANES, FEATURES))


def resets(num_steps: int = NUM_STEPS) -> jax.Array:
    return jnp.zeros((num_steps, NUM_LANES), bool)


def stepper() -> tuple[ResetRecurrence, Callable[..., Any]]:
    """A recurrence and its bound apply, so every call within one test shares its params."""
    recurrence = ResetRecurrence(hidden_size=HIDDEN)
    params = recurrence.init(jax.random.key(1), recurrence.initialize_carry(NUM_LANES), inputs(), resets())
    return recurrence, partial(recurrence.apply, params)


def test_it_applies_at_the_learn_time_layout() -> None:
    recurrence, step = stepper()

    carry, outputs = step(recurrence.initialize_carry(NUM_LANES), inputs(), resets())

    assert outputs.shape == (NUM_STEPS, NUM_LANES, HIDDEN)
    assert jax.tree.all(jax.tree.map(lambda leaf: leaf.shape == (NUM_LANES, HIDDEN), carry))


def test_a_reset_replaces_the_incoming_carry_with_the_freshly_initialised_one() -> None:
    recurrence, step = stepper()
    fresh = recurrence.initialize_carry(NUM_LANES)
    held = jax.tree.map(lambda leaf: leaf + 7.0, fresh)

    after_reset, _ = step(held, inputs(1), jnp.ones((1, NUM_LANES), bool))
    from_fresh, _ = step(fresh, inputs(1), resets(1))

    assert jnp.allclose(after_reset, from_fresh, atol=1e-6)


def test_scanning_a_sequence_equals_stepping_one_at_a_time() -> None:
    recurrence, step = stepper()
    sequence, flags = inputs(), resets().at[2, 0].set(True)

    scanned_carry, scanned = step(recurrence.initialize_carry(NUM_LANES), sequence, flags)

    carry, stepped = recurrence.initialize_carry(NUM_LANES), []
    for moment in range(NUM_STEPS):
        carry, output = step(carry, sequence[moment : moment + 1], flags[moment : moment + 1])
        stepped.append(output)

    assert jnp.allclose(scanned, jnp.concatenate(stepped), atol=1e-6)
    assert jnp.allclose(scanned_carry, carry, atol=1e-6)


def test_a_reset_on_one_lane_leaves_every_other_lane_undisturbed() -> None:
    recurrence, step = stepper()
    held = jax.tree.map(lambda leaf: leaf + 7.0, recurrence.initialize_carry(NUM_LANES))
    one_lane = resets(1).at[0, 0].set(True)

    reset_lane, _ = step(held, inputs(1), one_lane)
    nothing_reset, _ = step(held, inputs(1), resets(1))

    assert jnp.allclose(reset_lane[1:], nothing_reset[1:], atol=1e-6)
    assert not jnp.allclose(reset_lane[0], nothing_reset[0], atol=1e-6)


def test_any_cell_is_reset_including_one_whose_carry_is_a_tuple() -> None:
    """An LSTM carry is two arrays, so the reset has to reach every leaf of it."""
    recurrence = ResetRecurrence(hidden_size=HIDDEN, cell=nn.OptimizedLSTMCell)
    fresh = recurrence.initialize_carry(NUM_LANES)
    params = recurrence.init(jax.random.key(1), fresh, inputs(), resets())
    held = jax.tree.map(lambda leaf: leaf + 7.0, fresh)

    after_reset, _ = recurrence.apply(params, held, inputs(1), jnp.ones((1, NUM_LANES), bool))
    from_fresh, _ = recurrence.apply(params, fresh, inputs(1), resets(1))

    assert len(jax.tree.leaves(fresh)) == 2
    assert jax.tree.all(jax.tree.map(lambda a, b: jnp.allclose(a, b, atol=1e-6), after_reset, from_fresh))
