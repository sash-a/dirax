"""What recurrence adds to PPO: a carry that survives a step and dies at an episode boundary.

Three things have to agree for a recurrent run to be training on anything real — the assembly, the
input type that feeds it, and a minibatch that is still a set of sequences — so `recurrent_wiring`
builds all three together and these tests hold the agreement rather than any one piece of it.

The sharpest property is that acting and learning are the same network at different time widths: a
rollout records values from one step at a time, the loss recomputes them over a whole sequence, and
if those disagree then every stored log probability the ratio is built from is comparing two
different networks.
"""

from dataclasses import replace

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from brrrl.algorithms.ppo.ppo import (
    PPOUpdateRule,
    Sequences,
    Steps,
    feedforward_wiring,
    recurrent_wiring,
)
from brrrl.algorithms.ppo.types import PPOBatch, PPOPolicyInput, PPOTrajectory, PPOTransition, RecPPOPolicyInput

MOMENTS, ENVS, OBS, ACTIONS, HIDDEN = 6, 4, 5, 3, 16
SLOTS = MOMENTS + 1
MINIBATCHES = 2

OBS_SPACE = gym.spaces.Box(-1.0, 1.0, (OBS,), np.float32)
ACT_SPACE = gym.spaces.Discrete(ACTIONS)


def wired():
    """A recurrent wiring and its initial params, so every call in one test shares them."""
    wiring = recurrent_wiring(OBS_SPACE, ACT_SPACE, (ENVS,), HIDDEN)
    return wiring, wiring.init_params(jax.random.key(1))


def sequence() -> jax.Array:
    return jax.random.normal(jax.random.key(2), (MOMENTS, ENVS, OBS))


def quiet() -> jax.Array:
    """Flags for a stretch of interaction no episode ended in."""
    return jnp.zeros((MOMENTS, ENVS), bool)


def stamped() -> PPOBatch:
    """A batch whose every leaf reads ``sequence * 100 + moment``, so a cut can be read back."""
    marks = jnp.array([[s * 100 + t for t in range(MOMENTS)] for s in range(ENVS)], jnp.float32)
    return PPOBatch(PPOTransition(*(marks,) * 7), marks, marks)


def trajectory() -> PPOTrajectory:
    """One rollout's slots, shaped as a Roller submits them: ``(rollouts, time, envs, ...)``."""
    shape = (1, SLOTS, ENVS)
    key = jax.random.key(3)
    return PPOTrajectory(
        PPOTransition(
            observation=jax.random.normal(key, (*shape, OBS)),
            terminated=jnp.zeros(shape, bool).at[0, SLOTS - 2].set(True),
            truncated=jnp.zeros(shape, bool),
            reward=jax.random.normal(key, shape),
            action=jax.random.randint(key, shape, 0, ACTIONS),
            value=jax.random.normal(key, shape),
            log_prob=-jnp.ones(shape),
        )
    )


def rule_for(wiring) -> PPOUpdateRule:
    return PPOUpdateRule(
        wiring.apply_fn,
        wiring.init_params,
        optax.adam(1e-3),
        ENVS,
        MOMENTS,
        num_epochs=2,
        num_minibatches=MINIBATCHES,
        policy_input=wiring.policy_input,
        minibatcher=wiring.minibatcher,
    )


def test_stepping_the_act_path_matches_scanning_the_loss_path() -> None:
    """The one property the whole arrangement rests on. A value a rollout recorded one step at a
    time has to be the value the loss recomputes over the sequence, or the stored log probability
    the ratio is built from belongs to a different network than the one being updated."""
    wiring, params = wired()
    moments, flags = sequence(), quiet()

    _, scanned = wiring.apply_fn(params, wiring.policy_input.of(moments, flags, flags))

    carry, stepped = wiring.carry, []
    for moment in range(MOMENTS):
        output, carry = wiring.act_fn(
            params,
            wiring.policy_input.of(moments[moment], flags[moment], flags[moment]),
            jax.random.key(0),
            carry,
        )
        stepped.append(output.value)

    assert jnp.allclose(scanned, jnp.stack(stepped), atol=1e-5)


def test_a_moment_reaches_every_moment_after_it() -> None:
    """Otherwise the scan is running down the wrong axis and the memory is of nothing."""
    wiring, params = wired()
    moments, flags = sequence(), quiet()

    kept, _ = wiring.apply_fn(params, wiring.policy_input.of(moments, flags, flags))
    nudged, _ = wiring.apply_fn(params, wiring.policy_input.of(moments.at[0].add(5.0), flags, flags))

    assert not any(jnp.allclose(kept.logits[at], nudged.logits[at], atol=1e-6) for at in range(MOMENTS))


def test_nothing_before_a_done_reaches_past_it() -> None:
    """The carry is replaced on the way out of the step whose episode ended, so the moments after a
    boundary read as though the ones before it never happened."""
    wiring, params = wired()
    moments, flags = sequence(), quiet()
    boundary = flags.at[1].set(True)

    kept, _ = wiring.apply_fn(params, wiring.policy_input.of(moments, boundary, flags))
    nudged, _ = wiring.apply_fn(params, wiring.policy_input.of(moments.at[0].add(5.0), boundary, flags))

    unchanged = [bool(jnp.allclose(kept.logits[at], nudged.logits[at], atol=1e-6)) for at in range(MOMENTS)]
    assert unchanged == [False, False, True, True, True, True]


def test_sequences_keeps_every_sequence_whole_and_in_order() -> None:
    """A scan over a minibatch has to be a scan over moments one environment really lived through."""
    cut = Sequences().cut(jax.random.key(0), stamped(), MINIBATCHES)

    for minibatch in range(MINIBATCHES):
        for column in np.asarray(cut.advantages[minibatch]).T:
            assert len({int(mark // 100) for mark in column}) == 1, "a column mixes sequences"
            assert [int(mark % 100) for mark in column] == list(range(MOMENTS)), "a column is out of order"


def test_sequences_cuts_time_major_and_steps_cuts_time_away() -> None:
    """`ResetRecurrence` scans the leading axis, so a minibatch of sequences has to lead with time."""
    sequences = Sequences().cut(jax.random.key(0), stamped(), MINIBATCHES)
    steps = Steps().cut(jax.random.key(0), stamped(), MINIBATCHES)

    assert sequences.advantages.shape == (MINIBATCHES, MOMENTS, ENVS // MINIBATCHES)
    assert steps.advantages.shape == (MINIBATCHES, ENVS * MOMENTS // MINIBATCHES)


def test_each_minibatcher_pools_what_it_cuts() -> None:
    """What ``num_minibatches`` has to divide, which is the only thing `isvalid` needs off one."""
    assert Steps().pool(ENVS, MOMENTS) == ENVS * MOMENTS
    assert Sequences().pool(ENVS, MOMENTS) == ENVS


def test_sequences_loses_no_step_and_duplicates_none() -> None:
    cut = Sequences().cut(jax.random.key(0), stamped(), MINIBATCHES)

    assert sorted(np.asarray(cut.advantages).ravel().tolist()) == sorted(
        np.asarray(stamped().advantages).ravel().tolist()
    )


@pytest.mark.parametrize("num_minibatches", [ENVS + 1, ENVS - 1])
def test_a_sequence_pool_refuses_a_minibatch_count_it_cannot_divide(num_minibatches: int) -> None:
    """The divisibility a recurrent run needs is over sequences, not steps, so a count that was fine
    for a flat pool can be impossible here — and `isvalid` has to say so rather than reshape badly."""
    wiring, _ = wired()
    rule = rule_for(wiring)

    assert not replace(rule, num_minibatches=num_minibatches).isvalid()


def test_a_recurrent_update_descends_and_reports_finite_metrics() -> None:
    wiring, _ = wired()
    rule = rule_for(wiring)
    assert rule.isvalid()
    state = rule.init(jax.random.key(4))

    advanced, metrics = jax.jit(rule.update)(state, trajectory())

    assert all(bool(jnp.all(jnp.isfinite(jnp.asarray(value)))) for value in metrics.values())
    assert not jax.tree.all(
        jax.tree.map(lambda before, after: bool(jnp.array_equal(before, after)), state.params, advanced.params)
    )


def test_only_the_recurrent_wiring_brings_a_carry() -> None:
    """What `make_policy` reads to decide which Policy threads state, so a feed-forward run never
    grows a carry and a recurrent one can never be built without one."""
    carry = recurrent_wiring(OBS_SPACE, ACT_SPACE, (ENVS,), HIDDEN).carry

    assert feedforward_wiring(OBS_SPACE, ACT_SPACE).carry is None
    assert carry is not None and carry.shape == (ENVS, HIDDEN)


def test_each_wiring_names_the_input_its_assembly_reads() -> None:
    """The pairing that cannot be mixed: an input with no flags cannot reset a carry, and one with
    them has nothing to say to a feed-forward Torso."""
    assert feedforward_wiring(OBS_SPACE, ACT_SPACE).policy_input is PPOPolicyInput
    assert recurrent_wiring(OBS_SPACE, ACT_SPACE, (ENVS,), HIDDEN).policy_input is RecPPOPolicyInput


def test_an_input_reads_both_boundary_flags_as_one_done() -> None:
    """A carry does not care which way an episode ended, only that it did."""
    term = jnp.zeros((ENVS,), bool).at[0].set(True)
    trunc = jnp.zeros((ENVS,), bool).at[1].set(True)

    assert jnp.array_equal(RecPPOPolicyInput.of(jnp.zeros((ENVS, OBS)), term, trunc).dones, term | trunc)
