"""The reset discipline, at both layouts the module is applied at.

Acting is this module over a length-one time axis and learning is the same module over many, so
scanning a sequence and stepping it one observation at a time have to agree — that test is what
pins the act path to the learn path, and so what stops a Trajectory carrying states that never acted.
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from dirax.networks.recurrent import ResetRecurrence

NUM_STEPS = 5
NUM_ENVS = 4
NUM_AGENTS = 2
FEATURES = 6
HIDDEN = 8

ENVS = (NUM_ENVS,)
TEAM = (NUM_ENVS, NUM_AGENTS)
"""The two prefixes a stepper reports at: an env axis alone, and an env axis with a team on it."""


def inputs(num_steps: int = NUM_STEPS, prefix: tuple[int, ...] = ENVS) -> jax.Array:
    return jax.random.normal(jax.random.key(0), (num_steps, *prefix, FEATURES))


def dones(num_steps: int = NUM_STEPS, prefix: tuple[int, ...] = ENVS) -> jax.Array:
    return jnp.zeros((num_steps, *prefix), bool)


def stepper(
    prefix: tuple[int, ...] = ENVS,
    cell: Callable[..., nn.RNNCellBase] = nn.GRUCell,
) -> tuple[ResetRecurrence, Callable[..., Any]]:
    """A recurrence and its bound apply, so every call within one test shares its params."""
    recurrence = ResetRecurrence(hidden_size=HIDDEN, cell=cell)
    carry = recurrence.initialize_carry(prefix)
    params = recurrence.init(jax.random.key(1), carry, inputs(prefix=prefix), dones(prefix=prefix))
    return recurrence, partial(recurrence.apply, params)


def held(recurrence: ResetRecurrence, prefix: tuple[int, ...] = ENVS) -> Any:
    """A carry that is distinguishable from a freshly-initialised one, which is zeros."""
    return jax.tree.map(lambda leaf: leaf + 7.0, recurrence.initialize_carry(prefix))


def test_it_applies_at_the_learn_time_layout() -> None:
    recurrence, step = stepper()

    carry, outputs = step(recurrence.initialize_carry(ENVS), inputs(), dones())

    assert outputs.shape == (NUM_STEPS, NUM_ENVS, HIDDEN)
    assert jax.tree.all(jax.tree.map(lambda leaf: leaf.shape == (NUM_ENVS, HIDDEN), carry))


def test_it_applies_at_the_same_layout_with_an_agent_axis() -> None:
    """ADR-0012: the module says nothing about agents, it carries whatever leading axes it is handed,
    so the prefix it initialises a carry at is the one the stepper reported and nothing else."""
    recurrence, step = stepper(TEAM)

    carry, outputs = step(recurrence.initialize_carry(TEAM), inputs(prefix=TEAM), dones(prefix=TEAM))

    assert outputs.shape == (NUM_STEPS, NUM_ENVS, NUM_AGENTS, HIDDEN)
    assert jax.tree.all(jax.tree.map(lambda leaf: leaf.shape == (NUM_ENVS, NUM_AGENTS, HIDDEN), carry))


def test_a_done_replaces_the_outgoing_carry_with_the_freshly_initialised_one() -> None:
    """Next-step autoreset (ADR-0001) puts the final observation on the done step, so the carry is
    replaced on the way out — in time for the first observation of the next episode, not before the
    last one of this episode is read."""
    recurrence, step = stepper()

    after_done, _ = step(held(recurrence), inputs(1), jnp.ones((1, NUM_ENVS), bool))

    assert jnp.allclose(after_done, recurrence.initialize_carry(ENVS), atol=1e-6)


def test_a_done_leaves_the_output_of_its_own_step_alone() -> None:
    """The final observation belongs to the episode that is ending, and truncation bootstrapping
    reads its value, so it has to be embedded with that episode's memory rather than a blank one."""
    recurrence, step = stepper()

    _, ended = step(held(recurrence), inputs(1), jnp.ones((1, NUM_ENVS), bool))
    _, continued = step(held(recurrence), inputs(1), dones(1))

    assert jnp.allclose(ended, continued, atol=1e-6)


def test_no_memory_crosses_an_episode_boundary() -> None:
    """The step after a done reads as though nothing preceded it."""
    recurrence, step = stepper()
    sequence, flags = inputs(2), dones(2).at[0].set(True)

    _, across = step(held(recurrence), sequence, flags)
    _, afresh = step(recurrence.initialize_carry(ENVS), sequence[1:], dones(1))

    assert jnp.allclose(across[1], afresh[0], atol=1e-6)


def test_scanning_a_sequence_equals_stepping_one_at_a_time() -> None:
    """The act path is this module at one step and the loss is the same module at many, so a
    disagreement here is a Trajectory whose carries are not the ones its actions were taken under."""
    recurrence, step = stepper()
    sequence, flags = inputs(), dones().at[2, 0].set(True)

    scanned_carry, scanned = step(recurrence.initialize_carry(ENVS), sequence, flags)

    carry, stepped = recurrence.initialize_carry(ENVS), []
    for moment in range(NUM_STEPS):
        carry, output = step(carry, sequence[moment : moment + 1], flags[moment : moment + 1])
        stepped.append(output)

    assert jnp.allclose(scanned, jnp.concatenate(stepped), atol=1e-6)
    assert jnp.allclose(scanned_carry, carry, atol=1e-6)


def test_a_done_on_one_env_leaves_every_other_env_undisturbed() -> None:
    recurrence, step = stepper()
    one_env = dones(1).at[0, 0].set(True)

    done_env, _ = step(held(recurrence), inputs(1), one_env)
    nothing_done, _ = step(held(recurrence), inputs(1), dones(1))

    assert jnp.allclose(done_env[1:], nothing_done[1:], atol=1e-6)
    assert not jnp.allclose(done_env[0], nothing_done[0], atol=1e-6)


def test_a_done_on_one_agent_leaves_its_teammates_undisturbed() -> None:
    """An agent axis is another leading axis, so a reset lands per entry rather than per env."""
    recurrence, step = stepper(TEAM)
    one_agent = dones(1, TEAM).at[0, 0, 0].set(True)

    done_agent, _ = step(held(recurrence, TEAM), inputs(1, TEAM), one_agent)
    nothing_done, _ = step(held(recurrence, TEAM), inputs(1, TEAM), dones(1, TEAM))

    assert jnp.allclose(done_agent[1:], nothing_done[1:], atol=1e-6)
    assert jnp.allclose(done_agent[0, 1:], nothing_done[0, 1:], atol=1e-6)
    assert not jnp.allclose(done_agent[0, 0], nothing_done[0, 0], atol=1e-6)


def test_any_cell_is_reset_including_one_whose_carry_is_a_tuple() -> None:
    """An LSTM carry is two arrays, so the reset has to reach every leaf of it."""
    recurrence, step = stepper(cell=nn.OptimizedLSTMCell)
    fresh = recurrence.initialize_carry(ENVS)

    after_done, _ = step(held(recurrence), inputs(1), jnp.ones((1, NUM_ENVS), bool))

    assert len(jax.tree.leaves(fresh)) == 2
    assert jax.tree.all(jax.tree.map(lambda a, b: jnp.allclose(a, b, atol=1e-6), after_done, fresh))
