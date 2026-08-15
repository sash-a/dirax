"""What PPO's update rule must hold, most of it checked analytically rather than by regression.

Two properties carry the file. At the behaviour policy the importance ratio is exactly one, which
pins the surrogate, the approximate KL and the clip fraction to values that can be asserted rather
than recorded. And a Final obs slot's action and reward are meaningless (ADR-0001), so filling them
with garbage must not move a single metric — that is the whole reason the mask exists.

The Assembly here is a test double: ADR-0008 leaves Assembly to each algorithm, and PPO's runs the
critic beside the actor because its Roller needs a value at act time.
"""

from dataclasses import replace
from typing import cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from brrrl.algorithms.estimators import generalised_advantages
from brrrl.algorithms.ppo.ppo import PPOApply, PPOUpdateRule, explained_variance, normalise
from brrrl.algorithms.ppo.types import PPOBatch, PPOTrajectory, PPOTransition, PPOUpdateState
from brrrl.networks.head import DiscreteHead, ValueHead
from brrrl.networks.torso import MLPTorso

BATCH, ROLLOUT, OBS, ACTIONS = 4, 8, 5, 3
SLOTS = ROLLOUT + 1
EPOCHS, MINIBATCHES = 2, 2
TERMINATION, TRUNCATION = (0, 6), (1, 3)
"""(row, slot) of a Final obs of each kind, so both boundaries are exercised."""


class Assembly(nn.Module):
    """PPO's Assembly: a distribution and a value from one embedding."""

    num_actions: int

    @nn.compact
    def __call__(self, observation: jax.Array):
        embedding = MLPTorso(layer_sizes=(8,))(observation)
        return DiscreteHead(self.num_actions)(embedding), ValueHead()(embedding)


@pytest.fixture(scope="module")
def assembly() -> Assembly:
    return Assembly(ACTIONS)


@pytest.fixture(scope="module")
def params(assembly: Assembly):
    return assembly.init(jax.random.key(0), jnp.zeros((1, OBS)))


@pytest.fixture(scope="module")
def apply_fn(assembly: Assembly) -> PPOApply:
    """Flax's ``apply`` returns its mutated variables too when asked, so its type admits a tuple this
    seam never wants. Narrowed here; real wiring will need the same."""
    return cast(PPOApply, assembly.apply)


@pytest.fixture(scope="module")
def rule(apply_fn: PPOApply) -> PPOUpdateRule:
    return PPOUpdateRule(
        apply_fn=apply_fn,
        optimizer=optax.adam(1e-3),
        policy_clip_eps=0.2,
        value_clip_eps=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        batch_size=BATCH,
        rollout_length=ROLLOUT,
        num_epochs=EPOCHS,
        num_minibatches=MINIBATCHES,
        gamma=0.99,
        gae_lambda=0.95,
    )


@pytest.fixture
def trajectory(assembly: Assembly, params) -> PPOTrajectory:
    """Acted under ``params`` and read back to the host, which is how an Actor ships one.

    Host-resident on purpose: ``ArrayTree`` is numpy, and the ratio being exactly one below depends
    on these log-probabilities being the ones the policy actually acted under.
    """
    key = jax.random.key(1)
    observation = jax.random.normal(key, (BATCH, SLOTS, OBS))
    distribution, value = assembly.apply(params, observation)
    action = distribution.sample(seed=key)

    terminated, truncated = np.zeros((BATCH, SLOTS), bool), np.zeros((BATCH, SLOTS), bool)
    terminated[TERMINATION] = True
    truncated[TRUNCATION] = True
    reward = jax.random.normal(jax.random.key(2), (BATCH, SLOTS))
    return PPOTrajectory(
        transitions=PPOTransition(
            observation=np.asarray(observation),
            terminated=terminated,
            truncated=truncated,
            reward=np.asarray(reward),
            action=np.asarray(action),
            value=np.asarray(value),
            log_prob=np.asarray(distribution.log_prob(action)),
        ),
        version=np.zeros(1, np.float32),
    )


@pytest.fixture
def state(rule: PPOUpdateRule, params) -> PPOUpdateState:
    return PPOUpdateState(params, rule.optimizer.init(params), jax.random.key(3))


def test_the_ratio_is_one_at_the_behaviour_policy(rule: PPOUpdateRule, state, trajectory) -> None:
    """So the surrogate is exactly the mean advantage, which normalisation has already zeroed."""
    batch = PPOBatch(*from_trajectory(rule, trajectory))
    _, metrics = rule.loss(state.params, batch)

    assert np.isclose(metrics["approx_kl"], 0.0, atol=1e-6)
    assert np.isclose(metrics["clip_fraction"], 0.0, atol=1e-6)
    assert np.isclose(metrics["policy_loss"], 0.0, atol=1e-5)


def test_a_final_obs_slot_cannot_move_any_metric(rule: PPOUpdateRule, state, trajectory) -> None:
    """Its action and reward are the meaningless half of the slot; only its observation counts."""
    _, clean = rule.update(state, trajectory)

    transitions = trajectory.transitions
    final = transitions.terminated | transitions.truncated
    fouled = trajectory._replace(
        transitions=transitions._replace(
            reward=np.where(final, np.float32(1e4), transitions.reward),
            action=np.where(final, (np.asarray(transitions.action) + 1) % ACTIONS, transitions.action),
            log_prob=np.where(final, np.float32(-1e3), transitions.log_prob),
        )
    )
    _, garbage = rule.update(state, fouled)

    assert all(np.isclose(clean[key], garbage[key], atol=1e-5) for key in clean), {
        key: (float(clean[key]), float(garbage[key]))
        for key in clean
        if not np.isclose(clean[key], garbage[key], atol=1e-5)
    }


def test_update_advances_the_state_it_was_given(rule: PPOUpdateRule, state, trajectory) -> None:
    updated, _ = rule.update(state, trajectory)

    assert jax.tree.structure(updated.params) == jax.tree.structure(state.params)
    assert not jax.tree.all(jax.tree.map(jnp.allclose, state.params, updated.params))
    assert not jnp.array_equal(jax.random.key_data(updated.key), jax.random.key_data(state.key))


def test_metrics_are_scalars_the_collector_can_take(rule: PPOUpdateRule, state, trajectory) -> None:
    """A Metric recorder takes ``Mapping[str, ArrayLike]``; a per-minibatch axis would leak out here."""
    _, metrics = rule.update(state, trajectory)

    assert set(metrics) == {
        "total_loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "normalised_entropy",
        "approx_kl",
        "clip_fraction",
        "value_clip_fraction",
        "grad_norm",
        "update_norm",
        "param_norm",
        "explained_variance",
        "advantage_mean",
        "advantage_std",
        "advantage_positive_fraction",
        "mask_fraction",
        "approx_kl_first_minibatch",
        "approx_kl_last_minibatch",
    }
    assert all(np.ndim(value) == 0 for value in metrics.values())


def test_repeated_updates_descend(rule: PPOUpdateRule, state, trajectory) -> None:
    """The loop has to actually optimise, not merely run."""
    first = last = None
    for _ in range(8):
        state, metrics = rule.update(state, trajectory)
        first = metrics["value_loss"] if first is None else first
        last = metrics["value_loss"]

    assert last < first


def test_a_fresh_discrete_policy_is_near_uniform(rule: PPOUpdateRule, state, trajectory) -> None:
    """Entropy is read off the distribution rather than recomputed, so it starts at log(actions),
    which is the whole of a uniform policy's and so reads as one against it."""
    batch = PPOBatch(*from_trajectory(rule, trajectory))
    _, metrics = rule.loss(state.params, batch)

    assert np.isclose(metrics["entropy"], np.log(ACTIONS), atol=1e-2)
    assert np.isclose(metrics["normalised_entropy"], 1.0, atol=1e-2)


def test_the_update_brackets_its_own_walk_from_the_behaviour_policy(rule: PPOUpdateRule, state, trajectory) -> None:
    """The first minibatch is the only one taken before any step has moved the params, so its KL is
    zero outright rather than merely small. What the last one reads is the drift the whole point of
    ``num_epochs`` is to trade against, and a mean over every minibatch shows neither."""
    _, metrics = rule.update(state, trajectory)

    assert np.isclose(metrics["approx_kl_first_minibatch"], 0.0, atol=1e-6)
    assert metrics["approx_kl_last_minibatch"] > 0


def test_mask_fraction_counts_the_slots_the_loss_is_allowed(rule: PPOUpdateRule, state, trajectory) -> None:
    """One termination and one truncation, both inside the sliced batch."""
    _, metrics = rule.update(state, trajectory)

    assert np.isclose(metrics["mask_fraction"], (BATCH * ROLLOUT - 2) / (BATCH * ROLLOUT))


def test_each_clip_epsilon_bounds_only_its_own_term(rule: PPOUpdateRule, state, trajectory) -> None:
    """One bounds a dimensionless ratio, the other a drift in return units. Sharing one epsilon ties
    how hard the critic is clipped to whatever scale the environment's rewards happen to have."""
    moved, _ = rule.update(state, trajectory)
    batch = PPOBatch(*from_trajectory(rule, trajectory))

    _, tight = replace(rule, value_clip_eps=1e-8).loss(moved.params, batch)
    _, loose = replace(rule, value_clip_eps=1e8).loss(moved.params, batch)

    assert np.isclose(tight["value_clip_fraction"], 1.0)
    assert np.isclose(loose["value_clip_fraction"], 0.0)
    assert tight["clip_fraction"] == loose["clip_fraction"]


def test_the_value_clip_cannot_bind_at_the_behaviour_critic(rule: PPOUpdateRule, state, trajectory) -> None:
    """These values are the ones the critic itself produced, so there is nothing for it to drift from."""
    batch = PPOBatch(*from_trajectory(rule, trajectory))
    _, metrics = rule.loss(state.params, batch)

    assert np.isclose(metrics["value_clip_fraction"], 0.0)


def test_the_step_taken_is_reported_beside_the_gradient_that_asked_for_it(rule, state, trajectory) -> None:
    """Adam rescales per parameter, so what moved the weights is not the learning rate times the
    gradient, and only the step's own norm says how far they actually travelled."""
    _, slow = rule.update(state, trajectory)
    quick = replace(rule, optimizer=optax.adam(1e-1))
    _, fast = quick.update(state._replace(opt_state=quick.optimizer.init(state.params)), trajectory)

    assert slow["grad_norm"] > 0 and slow["param_norm"] > 0
    assert fast["update_norm"] > slow["update_norm"]


def test_shuffling_loses_no_step_and_duplicates_none(rule: PPOUpdateRule, trajectory) -> None:
    labelled = jnp.arange(BATCH * ROLLOUT, dtype=jnp.float32).reshape(BATCH, ROLLOUT)
    batch = PPOBatch(jax.tree.map(lambda leaf: leaf[:, :-1], trajectory.transitions), labelled, labelled)

    minibatches = rule.shuffle_and_minibatch(jax.random.key(4), batch)

    observation = jax.tree.leaves(minibatches.transitions.observation)[0]

    assert minibatches.advantages.shape == (MINIBATCHES, BATCH * ROLLOUT // MINIBATCHES)
    assert observation.shape == (MINIBATCHES, BATCH * ROLLOUT // MINIBATCHES, OBS)
    assert sorted(np.asarray(minibatches.advantages).ravel().tolist()) == list(range(BATCH * ROLLOUT))


def test_shuffling_keeps_each_step_whole(rule: PPOUpdateRule, trajectory) -> None:
    """A step's advantage must land beside its own transition, or the permutation has torn rows apart."""
    transitions = jax.tree.map(lambda leaf: leaf[:, :-1], trajectory.transitions)
    batch = PPOBatch(transitions, transitions.value, transitions.value)

    minibatches = rule.shuffle_and_minibatch(jax.random.key(5), batch)

    assert np.allclose(minibatches.advantages, minibatches.transitions.value)


@pytest.mark.parametrize(
    ("predictions", "expected"),
    [
        pytest.param(lambda targets: targets, 1.0, id="perfect"),
        pytest.param(lambda targets: jnp.full_like(targets, targets[:4].mean()), 0.0, id="predicts the mean"),
    ],
)
def test_explained_variance_is_anchored_at_its_two_reference_points(predictions, expected) -> None:
    targets = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    mask = jnp.asarray([1, 1, 1, 1, 0, 0], bool)

    assert np.isclose(explained_variance(targets, predictions(targets), mask), expected, atol=1e-5)


def test_explained_variance_ignores_the_slots_it_is_masked_off(rule: PPOUpdateRule) -> None:
    targets = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    mask = jnp.asarray([1, 1, 1, 1, 0, 0], bool)
    fouled = jnp.asarray([1.0, 2.0, 3.0, 4.0, 1e6, -1e6])

    assert np.isclose(explained_variance(targets, fouled, mask), 1.0, atol=1e-5)


def test_normalise_standardises_over_the_unmasked_slots_alone() -> None:
    values = jnp.asarray([1.0, 2.0, 3.0, 4.0, 1e6, -1e6])
    mask = jnp.asarray([1, 1, 1, 1, 0, 0], bool)

    normalised = normalise(values, mask)

    assert np.isclose(jnp.mean(normalised, where=mask), 0.0, atol=1e-5)
    assert np.isclose(jnp.std(normalised, where=mask), 1.0, atol=1e-4)


def test_the_rule_is_hashable_so_it_can_be_a_static_jit_argument(rule: PPOUpdateRule) -> None:
    """``update`` closes over ``self`` as static; an unfrozen dataclass would raise at trace time."""
    assert hash(rule) == hash(rule)


def from_trajectory(rule: PPOUpdateRule, trajectory: PPOTrajectory):
    """The batch ``update`` builds, rebuilt here so ``loss`` can be asserted on its own."""
    transitions = trajectory.transitions
    advantages = generalised_advantages(
        transitions.reward,
        transitions.value,
        transitions.terminated,
        transitions.truncated,
        rule.gamma,
        rule.gae_lambda,
        axis=1,
    )
    return jax.tree.map(lambda leaf: leaf[:, :-1], transitions), advantages, advantages + transitions.value[:, :-1]
