"""Recurrent PPO's assembly, and the property that makes its Trajectories mean anything.

Acting applies `RecPPONetwork` over a length-one time axis and the loss applies the same network
over a whole sequence. If those two disagree then every log probability a rollout stored belongs to
a different network than the one the ratio compares it against, and nothing else here would notice.

The assembly is the driver's own rather than a double, so what these hold for is what `main` runs.
"""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax import tree

from brrrl.algorithms.ppo.ppo import PPOUpdateRule
from brrrl.algorithms.ppo.recurrent import RecPPONetwork, make_network
from brrrl.algorithms.ppo.types import PPOTrajectory, PPOTransition, RecPPOPolicyInput, RecPPOTrajectory
from brrrl.learning.minibatcher import Sequences
from brrrl.networks.head import DiscreteHead, MaskedDiscreteHead
from brrrl.types import Params

MOMENTS, ENVS, OBS, ACTIONS, HIDDEN = 6, 4, 5, 3, 16
SLOTS = MOMENTS + 1
MINIBATCHES = 2

OBS_SPACE = gym.spaces.Box(-1.0, 1.0, (OBS,), np.float32)
ACT_SPACE = gym.spaces.Discrete(ACTIONS)


def network() -> RecPPONetwork:
    return make_network(OBS_SPACE, ACT_SPACE, HIDDEN)


def sequence() -> jax.Array:
    return jax.random.normal(jax.random.key(2), (MOMENTS, ENVS, OBS))


def quiet() -> jax.Array:
    """Flags for a stretch of interaction no episode ended in."""
    return jnp.zeros((MOMENTS, ENVS), bool)


def initialised(assembly: RecPPONetwork) -> Params:
    """Traced on the learn-time layout, which is the one with a time axis for the scan."""
    example = RecPPOPolicyInput(jnp.zeros((1, ENVS, OBS)), jnp.zeros((1, ENVS), bool))
    return assembly.init(jax.random.key(1), assembly.initialize_carry((ENVS,)), example)


def over_sequence(assembly: RecPPONetwork, params: Params, moments: jax.Array, dones: jax.Array, per_rollout=None):
    """The loss's forward pass: the whole sequence at once, from the carry recorded for it."""
    carry = assembly.initialize_carry(dones.shape[1:]) if per_rollout is None else per_rollout
    return assembly.apply(params, carry, RecPPOPolicyInput(moments, dones))


def learner_forward(assembly: RecPPONetwork):
    """What `main` hands the rule: the sequence pass with its carry hidden behind it."""

    def apply_fn(params: Params, policy_input: RecPPOPolicyInput, per_rollout):
        _, distribution, value = over_sequence(
            assembly, params, policy_input.observation, policy_input.dones, per_rollout
        )
        return distribution, value

    return apply_fn


def trajectory() -> RecPPOTrajectory:
    """One rollout's slots, shaped as a Roller submits them: ``(rollouts, time, envs, ...)``.

    The carry takes the width-one time axis a single-item timescale carries, so `batch_major` and
    `per_rollout` reduce it to the ``(sequences, ...)`` a minibatch is cut at.
    """
    shape = (1, SLOTS, ENVS)
    key = jax.random.key(3)
    return RecPPOTrajectory(
        PPOTransition(
            observation=jax.random.normal(key, (*shape, OBS)),
            terminated=jnp.zeros(shape, bool).at[0, SLOTS - 2].set(True),
            truncated=jnp.zeros(shape, bool),
            reward=jax.random.normal(key, shape),
            action=jax.random.randint(key, shape, 0, ACTIONS),
            value=jax.random.normal(key, shape),
            log_prob=-jnp.ones(shape),
        ),
        hidden_state=jax.random.normal(key, (1, 1, ENVS, HIDDEN)),
    )


def rule_for(assembly: RecPPONetwork, num_minibatches: int = MINIBATCHES) -> PPOUpdateRule:
    return PPOUpdateRule(
        learner_forward(assembly),
        lambda key: initialised(assembly),
        optax.adam(1e-3),
        ENVS,
        MOMENTS,
        num_epochs=2,
        num_minibatches=num_minibatches,
        policy_input=RecPPOPolicyInput,
        minibatcher=Sequences(),
    )


def test_stepping_one_moment_at_a_time_matches_scanning_the_sequence() -> None:
    """The property the whole arrangement rests on. A value a rollout recorded one step at a time
    has to be the value the loss recomputes over the sequence, or the stored log probability the
    ratio is built from belongs to a different network than the one being updated."""
    assembly, moments, flags = network(), sequence(), quiet()
    params = initialised(assembly)

    _, _, scanned = over_sequence(assembly, params, moments, flags)

    carry, stepped = assembly.initialize_carry((ENVS,)), []
    for at in range(MOMENTS):
        timed = tree.map(lambda leaf: leaf[None], RecPPOPolicyInput(moments[at], flags[at]))
        carry, _, value = assembly.apply(params, carry, timed)
        stepped.append(value[0])

    assert jnp.allclose(scanned, jnp.stack(stepped), atol=1e-5)


def test_a_moment_reaches_every_moment_after_it() -> None:
    """Otherwise the scan is running down the wrong axis and the memory is of nothing."""
    assembly, moments, flags = network(), sequence(), quiet()
    params = initialised(assembly)

    _, kept, _ = over_sequence(assembly, params, moments, flags)
    _, nudged, _ = over_sequence(assembly, params, moments.at[0].add(5.0), flags)

    assert not any(jnp.allclose(kept.logits[at], nudged.logits[at], atol=1e-6) for at in range(MOMENTS))


def test_nothing_before_a_done_reaches_past_it() -> None:
    """The carry is replaced on the way out of the step whose episode ended, so the moments after a
    boundary read as though the ones before it never happened."""
    assembly, moments, flags = network(), sequence(), quiet()
    params = initialised(assembly)
    boundary = flags.at[1].set(True)

    _, kept, _ = over_sequence(assembly, params, moments, boundary)
    _, nudged, _ = over_sequence(assembly, params, moments.at[0].add(5.0), boundary)

    unchanged = [bool(jnp.allclose(kept.logits[at], nudged.logits[at], atol=1e-6)) for at in range(MOMENTS)]
    assert unchanged == [False, False, True, True, True, True]


def test_a_carry_is_held_at_the_prefix_a_stepper_reports() -> None:
    """ADR-0012: the assembly is told a prefix, never an agent count, so a team's carry is per agent."""
    assembly = network()

    assert assembly.initialize_carry((ENVS,)).shape == (ENVS, HIDDEN)
    assert assembly.initialize_carry((ENVS, 3)).shape == (ENVS, 3, HIDDEN)


def test_the_carry_comes_back_ready_for_the_moment_after_the_one_it_ran() -> None:
    """What lets a `RecPolicy` hold it without knowing which flags to apply first."""
    assembly = network()
    params = initialised(assembly)
    fresh = assembly.initialize_carry((ENVS,))
    ended = jnp.ones((1, ENVS), bool)

    after_done, _, _ = assembly.apply(params, fresh, RecPPOPolicyInput(sequence()[:1], ended))

    assert jnp.allclose(after_done, fresh, atol=1e-6)


def test_a_recurrent_update_moves_the_params_and_reports_finite_metrics() -> None:
    assembly = network()
    rule = rule_for(assembly)
    assert rule.isvalid()
    state = rule.init(jax.random.key(4))

    advanced, metrics = jax.jit(rule.update)(state, trajectory())

    assert all(bool(jnp.all(jnp.isfinite(jnp.asarray(value)))) for value in metrics.values())
    moved = jax.tree.map(lambda before, after: bool(jnp.array_equal(before, after)), state.params, advanced.params)
    assert not jax.tree.all(moved)


@pytest.mark.parametrize("num_minibatches", [ENVS + 1, ENVS - 1])
def test_a_rule_over_sequences_refuses_a_minibatch_count_it_cannot_divide(num_minibatches: int) -> None:
    """The divisibility a recurrent run needs is over sequences, not steps, so a count that was fine
    for a flat pool can be impossible here — and `isvalid` has to say so rather than reshape badly."""
    assert not rule_for(network(), num_minibatches).isvalid()


def test_an_input_reads_both_boundary_flags_as_one_done() -> None:
    """A carry does not care which way an episode ended, only that it did."""
    term = jnp.zeros((ENVS,), bool).at[0].set(True)
    trunc = jnp.zeros((ENVS,), bool).at[1].set(True)

    assert jnp.array_equal(RecPPOPolicyInput.of(jnp.zeros((ENVS, OBS)), term, trunc).dones, term | trunc)


@pytest.mark.parametrize(
    ("obs_space", "expected"),
    [
        (OBS_SPACE, DiscreteHead),
        (
            gym.spaces.Dict({"observation": OBS_SPACE, "action_mask": gym.spaces.MultiBinary(ACTIONS)}),
            MaskedDiscreteHead,
        ),
    ],
    ids=["plain", "masked"],
)
def test_the_recurrent_assembly_picks_a_head_the_way_the_feed_forward_one_does(obs_space, expected) -> None:
    """The Head choice is shared with `ppo.make_head`, so recurrence changes nothing about it."""
    assert type(make_network(obs_space, ACT_SPACE).policy) is expected


def test_the_loss_reads_the_carry_the_trajectory_recorded() -> None:
    """The point of recording it. Two trajectories identical but for the carry their first slot
    acted under must not produce the same gradient — if they do, the loss is still starting every
    sequence from a blank memory and the recording is decoration."""
    assembly = network()
    rule = rule_for(assembly)
    state = rule.init(jax.random.key(4))
    recorded = trajectory()
    blank = recorded._replace(hidden_state=jnp.zeros_like(recorded.hidden_state))

    from_recorded, _ = jax.jit(rule.update)(state, recorded)
    from_blank, _ = jax.jit(rule.update)(state, blank)

    same = jax.tree.map(lambda a, b: bool(jnp.allclose(a, b, atol=1e-8)), from_recorded.params, from_blank.params)
    assert not jax.tree.all(same)


def test_a_trajectory_reports_the_carry_its_first_slot_acted_under() -> None:
    """``per_rollout`` drops the width-one time axis a single-item timescale carries, leaving the
    ``(sequences, ...)`` a minibatch is cut at."""
    recorded = trajectory()

    assert recorded.hidden_state.shape == (1, 1, ENVS, HIDDEN)
    assert recorded.per_rollout().shape == (1, ENVS, HIDDEN)


def test_a_feed_forward_trajectory_opens_with_nothing() -> None:
    """So the same update rule runs both, and `Steps` never sees an per_rollout it cannot cut."""
    assert PPOTrajectory(trajectory().transitions).per_rollout() == ()
