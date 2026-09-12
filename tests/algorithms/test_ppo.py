"""What PPO's update rule must hold, most of it checked analytically rather than by regression.

Two properties carry the file. At the behaviour policy the importance ratio is exactly one, which
pins the surrogate, the approximate KL and the clip fraction to values that can be asserted rather
than recorded. And a Final obs slot's action and reward are meaningless (ADR-0001), so filling them
with garbage must not move a single metric — that is the whole reason the mask exists.

Every one of them runs over both kinds of environment PPO builds an Assembly for, because the update
rule is meant to read the same whichever Head that Assembly ends in: a continuous action carries an
axis of its own and a log probability summed over it, a masked one arrives beside its observation,
and nothing else about the loss changes.

The Assemblies are PPO's own rather than doubles, so what these properties hold for is what `main`
runs. ADR-0008 leaves Assembly to each algorithm, and PPO's runs the critic beside the actor
because its Roller needs a value at act time.
"""

from collections.abc import Callable
from dataclasses import replace
from typing import Any, NamedTuple, cast

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from brrrl.acting.env_stepper import prefix
from brrrl.algorithms.estimators import generalised_advantages
from brrrl.algorithms.ppo.ppo import PPOApply, PPONetwork, PPOUpdateRule, explained_variance, make_network, normalise
from brrrl.algorithms.ppo.types import PPOBatch, PPOTrajectory, PPOTransition, PPOUpdateState, example
from brrrl.networks.head import ContinuousHead, DiscreteHead, MaskedDiscreteHead
from brrrl.types import TRANSITIONS, Array, Params, PRNGKey
from brrrl.wrappers.action_mask import ACTION_MASK, OBSERVATION

BATCH, ROLLOUT, OBS, ACTIONS = 4, 8, 5, 3
SLOTS = ROLLOUT + 1
EPOCHS, MINIBATCHES = 2, 2
TEAM = 3
"""Held at one agent and at a team, because the update rule is not supposed to be able to tell them
apart: whatever leading axes a field arrives with, nothing in the loss reads how many there are."""
TERMINATION, TRUNCATION = (0, 6), (1, 3)
"""(row, slot) of a Final obs of each kind, so both boundaries are exercised. It indexes a whole
agent row where there is one, because an episode is over when every agent is."""
ILLEGAL = ACTIONS - 1
"""Masked off everywhere in the masked case, so every property below is held over a mask that binds."""

PLAIN_SPACE = gym.spaces.Box(-1.0, 1.0, (OBS,))


def legal_actions(shape: tuple[int, ...]) -> np.ndarray:
    """A row of legal actions wherever the observation had one, which is the shape a Head reads its
    logits at. It follows the observation rather than being told a count, so a team's mask gains the
    agent axis from the observation it rides beside and a single agent's never has one."""
    legal = np.array([action != ILLEGAL for action in range(ACTIONS)], np.int8)
    return np.ones((*shape, ACTIONS), np.int8) * legal


class Space(NamedTuple):
    """One environment's half of what a shared test needs: the spaces PPO builds an Assembly from,
    an observation of the shape that Assembly reads, and a different action to write into a slot,
    which is a neighbouring category or a mirrored vector."""

    observation_space: gym.Space
    action_space: gym.Space
    observation: Callable[[jax.Array], Any]
    foul: Callable[[Array], Array]
    agent_axis: tuple[int, ...]


def spaces(num_agents: int) -> dict[str, Space]:
    """The three environments PPO builds an Assembly for, at whatever the agent axis is.

    A team's env observes for each of its agents, so the axis is on the observation and the mask
    alike and the network is handed it rather than producing it. A single agent's env has no agent
    axis on anything, which is what leaves its network with nothing to say about agents (ADR-0012).
    """
    axis = () if num_agents == 1 else (num_agents,)
    plain = gym.spaces.Box(-1.0, 1.0, (*axis, OBS))
    masked = gym.spaces.Dict({OBSERVATION: plain, ACTION_MASK: gym.spaces.MultiBinary((*axis, ACTIONS))})
    return {
        "discrete": Space(plain, gym.spaces.Discrete(ACTIONS), lambda array: array, lambda a: (a + 1) % ACTIONS, axis),
        "masked": Space(
            masked,
            gym.spaces.Discrete(ACTIONS),
            lambda array: {OBSERVATION: array, ACTION_MASK: legal_actions(array.shape[:-1])},
            lambda a: (a + 1) % ACTIONS,
            axis,
        ),
        "continuous": Space(plain, gym.spaces.Box(-1.0, 1.0, (ACTIONS,)), lambda array: array, lambda a: -a, axis),
    }


def shaped(space: Space, *leading: int) -> tuple[int, ...]:
    """The shape one of this env's observations arrives in, under ``leading`` axes of a caller's."""
    return (*leading, *space.agent_axis, OBS)


SPACES = spaces(1)
EVERY = {
    f"{name}{'' if agents == 1 else ' as a team'}": space
    for agents in (1, TEAM)
    for name, space in spaces(agents).items()
}


def initialiser(space: Space, network: nn.Module) -> Callable[[PRNGKey], Params]:
    """What a rule takes in place of an obs space: an Assembly paired with an example to trace it
    on. A network is shape-polymorphic, so neither half means `init` without the other."""
    return lambda key: network.init(key, space.observation(jnp.zeros(shaped(space, 1))))


def rule_over(space: Space, network: nn.Module) -> PPOUpdateRule:
    """The rule under test, over whichever Assembly it is handed."""
    return PPOUpdateRule(
        apply_fn=cast(PPOApply, network.apply),
        init_params=initialiser(space, network),
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


def acted(space: Space, network: nn.Module, params) -> PPOTrajectory:
    """A Trajectory acted under ``params`` and read back to the host, which is how an Actor ships one.

    Host-resident on purpose: ``ArrayTree`` is numpy, and the ratio being exactly one below depends
    on these log-probabilities being the ones the policy actually acted under.
    """
    key = jax.random.key(1)
    observation = space.observation(jax.random.normal(key, shaped(space, BATCH, SLOTS)))
    distribution, value = network.apply(params, observation)
    action = distribution.sample(seed=key)

    width = (BATCH, SLOTS, *space.agent_axis)
    terminated, truncated = np.zeros(width, bool), np.zeros(width, bool)
    terminated[TERMINATION] = True
    truncated[TRUNCATION] = True
    reward = jax.random.normal(jax.random.key(2), width)
    return PPOTrajectory(
        transitions=PPOTransition(
            observation=jax.tree.map(np.asarray, observation),
            terminated=terminated,
            truncated=truncated,
            reward=np.asarray(reward),
            action=np.asarray(action),
            value=np.asarray(value),
            log_prob=np.asarray(distribution.log_prob(action)),
        ),
    )


@pytest.fixture(scope="module", params=list(EVERY.values()), ids=list(EVERY))
def space(request: pytest.FixtureRequest) -> Space:
    """Every action space PPO has a Head for, at both widths of the agent axis, threaded through
    every fixture below."""
    return request.param


@pytest.fixture(scope="module")
def network(space: Space) -> nn.Module:
    return make_network(space.observation_space, space.action_space)


@pytest.fixture(scope="module")
def params(space: Space, network: nn.Module):
    return initialiser(space, network)(jax.random.key(0))


@pytest.fixture(scope="module")
def apply_fn(network: nn.Module) -> PPOApply:
    """Flax's ``apply`` returns its mutated variables too when asked, so its type admits a tuple this
    seam never wants. Narrowed here; real wiring will need the same."""
    return cast(PPOApply, network.apply)


@pytest.fixture(scope="module")
def rule(space: Space, network: nn.Module) -> PPOUpdateRule:
    return rule_over(space, network)


def stacked(trajectory: PPOTrajectory) -> PPOTrajectory:
    """``acted`` as a batch of rollouts reaches ``update``: one env per rollout.

    ``acted`` builds the ``(batch, time, ...)`` the update works in and every property here reads,
    which is one merge short of what an update is handed. Widening it back is what lets both live in
    one fixture: `brrrl.algorithms.ppo.ppo.batch_major` collapses this env axis straight back out.
    """
    return jax.tree.map(lambda leaf: leaf[:, :, None], trajectory)


@pytest.fixture
def trajectory(space: Space, network: nn.Module, params) -> PPOTrajectory:
    return acted(space, network, params)


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


def test_a_final_obs_slot_cannot_move_any_metric(rule: PPOUpdateRule, state, trajectory, space: Space) -> None:
    """Its action and reward are the meaningless half of the slot; only its observation counts."""
    _, clean = rule.update(state, stacked(trajectory))

    transitions = trajectory.transitions
    final = transitions.terminated | transitions.truncated
    action = np.array(transitions.action)
    action[final] = space.foul(action[final])  # Indexed, not ``where``: a continuous action has an axis.
    fouled = trajectory._replace(
        transitions=transitions._replace(
            reward=np.where(final, np.float32(1e4), transitions.reward),
            action=action,
            log_prob=np.where(final, np.float32(-1e3), transitions.log_prob),
        )
    )
    _, garbage = rule.update(state, stacked(fouled))

    assert all(np.isclose(clean[key], garbage[key], atol=1e-5) for key in clean), {
        key: (float(np.asarray(clean[key])), float(np.asarray(garbage[key])))
        for key in clean
        if not np.isclose(clean[key], garbage[key], atol=1e-5)
    }


def test_update_advances_the_state_it_was_given(rule: PPOUpdateRule, state, trajectory) -> None:
    updated, _ = rule.update(state, stacked(trajectory))

    assert jax.tree.structure(updated.params) == jax.tree.structure(state.params)
    assert not jax.tree.all(jax.tree.map(jnp.allclose, state.params, updated.params))
    assert not jnp.array_equal(jax.random.key_data(updated.key), jax.random.key_data(state.key))


def test_metrics_are_scalars_the_collector_can_take(rule: PPOUpdateRule, state, trajectory) -> None:
    """A Metric recorder takes ``Mapping[str, ArrayLike]``; a per-minibatch axis would leak out here."""
    _, metrics = rule.update(state, stacked(trajectory))

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
        state, metrics = rule.update(state, stacked(trajectory))
        first = metrics["value_loss"] if first is None else first
        last = metrics["value_loss"]

    assert np.asarray(last) < np.asarray(first)


@pytest.mark.parametrize(
    ("name", "legal"),
    [pytest.param("discrete", ACTIONS, id="discrete"), pytest.param("masked", ACTIONS - 1, id="masked")],
)
def test_a_fresh_discrete_policy_is_near_uniform_over_the_actions_left_to_it(name: str, legal: int) -> None:
    """Entropy is read off the distribution rather than recomputed, so it starts at the log of how
    many actions are legal — the whole of a uniform policy's, and so one against it. Masking an
    action lowers the first of those and the second not at all, which is the point of normalising."""
    _, metrics = fresh_loss(SPACES[name])

    assert np.isclose(metrics["entropy"], np.log(legal), atol=1e-2)
    assert np.isclose(metrics["normalised_entropy"], 1.0, atol=1e-2)


def test_a_fresh_continuous_policy_is_wide_without_reaching_uniform() -> None:
    """Its scale starts at a softplus of nearly nothing, which is broad over ``(-1, 1)`` without being
    the uniform policy on it — no squashed Gaussian is that, so the metric reads below one and the
    entropy bonus has somewhere left to push."""
    _, metrics = fresh_loss(SPACES["continuous"])

    assert 0.8 < metrics["normalised_entropy"] < 1.0


def test_the_update_brackets_its_own_walk_from_the_behaviour_policy(rule: PPOUpdateRule, state, trajectory) -> None:
    """The first minibatch is the only one taken before any step has moved the params, so its KL is
    zero outright rather than merely small. What the last one reads is the drift the whole point of
    ``num_epochs`` is to trade against, and a mean over every minibatch shows neither."""
    _, metrics = rule.update(state, stacked(trajectory))

    assert np.isclose(metrics["approx_kl_first_minibatch"], 0.0, atol=1e-6)
    assert np.asarray(metrics["approx_kl_last_minibatch"]) > 0


def test_mask_fraction_counts_the_slots_the_loss_is_allowed(rule: PPOUpdateRule, state, trajectory) -> None:
    """One termination and one truncation, both inside the sliced batch."""
    _, metrics = rule.update(state, stacked(trajectory))

    assert np.isclose(metrics["mask_fraction"], (BATCH * ROLLOUT - 2) / (BATCH * ROLLOUT))


def test_each_clip_epsilon_bounds_only_its_own_term(rule: PPOUpdateRule, state, trajectory) -> None:
    """One bounds a dimensionless ratio, the other a drift in return units. Sharing one epsilon ties
    how hard the critic is clipped to whatever scale the environment's rewards happen to have."""
    moved, _ = rule.update(state, stacked(trajectory))
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
    _, slow = rule.update(state, stacked(trajectory))
    quick = replace(rule, optimizer=optax.adam(1e-1))
    _, fast = quick.update(state._replace(opt_state=quick.optimizer.init(state.params)), stacked(trajectory))

    assert slow["grad_norm"] > 0 and slow["param_norm"] > 0
    assert fast["update_norm"] > slow["update_norm"]


def test_shuffling_loses_no_step_and_duplicates_none(rule: PPOUpdateRule, trajectory) -> None:
    labelled = jnp.arange(BATCH * ROLLOUT, dtype=jnp.float32).reshape(BATCH, ROLLOUT)
    batch = PPOBatch(jax.tree.map(lambda leaf: leaf[:, :-1], trajectory.transitions), labelled, labelled)

    minibatches = rule.shuffle_and_minibatch(jax.random.key(4), batch)

    minibatch_size = BATCH * ROLLOUT // MINIBATCHES

    assert minibatches.advantages.shape == (MINIBATCHES, minibatch_size)
    # Whatever a leaf carries of its own — a continuous action's axis, a mask's, an observation's —
    # is trailing context rather than a step to cut on, so only the two leading axes are the cut.
    assert all(
        cut.shape == (MINIBATCHES, minibatch_size, *whole.shape[2:])
        for cut, whole in zip(jax.tree.leaves(minibatches.transitions), jax.tree.leaves(trajectory.transitions))
    )
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


def test_a_value_carries_whatever_axis_its_observation_did(space: Space, network, params) -> None:
    """The network says nothing about agents and needs to: every layer in it is polymorphic over its
    leading axes, so a team's observation comes back as a value per agent and a single agent's as one
    value — which is the shape the loss needs either way, because it compares a value against a
    reward slot for slot and the stepper reported both the same way."""
    distribution, value = network.apply(params, space.observation(jnp.zeros(shaped(space, BATCH, SLOTS))))
    action = distribution.sample(seed=jax.random.key(6))

    assert value.shape == (BATCH, SLOTS, *space.agent_axis)
    assert jnp.shape(distribution.log_prob(action)) == (BATCH, SLOTS, *space.agent_axis)


def test_one_network_serves_both_arities_unchanged() -> None:
    """The property the whole arrangement is for: the same `PPONetwork`, built the same way, run over
    a single agent's observation and a team's. Nothing about it was told how many agents there are,
    and the params are the same size either way — one network shared down the axis, not N of them."""
    alone, team = spaces(1)["discrete"], spaces(TEAM)["discrete"]
    network = make_network(alone.observation_space, alone.action_space)
    apply = cast(PPOApply, network.apply)

    one = network.init(jax.random.key(0), jnp.zeros(shaped(alone, BATCH)))
    many = network.init(jax.random.key(0), jnp.zeros(shaped(team, BATCH)))

    assert jax.tree.all(jax.tree.map(lambda a, b: a.shape == b.shape, one, many))
    _, single_value = apply(one, np.zeros(shaped(alone, BATCH), np.float32))
    _, team_value = apply(many, np.zeros(shaped(team, BATCH), np.float32))
    assert single_value.shape == (BATCH,)
    assert team_value.shape == (BATCH, TEAM)


def test_the_rule_is_hashable_so_it_can_be_a_static_jit_argument(rule: PPOUpdateRule) -> None:
    """``update`` closes over ``self`` as static; an unfrozen dataclass would raise at trace time."""
    assert hash(rule) == hash(rule)


HEADS = {"discrete": DiscreteHead, "masked": MaskedDiscreteHead, "continuous": ContinuousHead}


@pytest.mark.parametrize(("name", "space"), list(SPACES.items()), ids=list(SPACES))
def test_the_head_follows_the_spaces(name: str, space: Space) -> None:
    """The whole of what an Assembly reads off an environment, and the only place PPO branches on it.

    The action space picks a Head; whether the observation carries a mask picks between the two
    discrete ones, so an env that masks nothing never pays for a mask.
    """
    head = make_network(space.observation_space, space.action_space).policy

    assert type(head) is HEADS[name]
    assert head.num_actions == ACTIONS


def test_a_head_is_built_from_one_agents_action_space() -> None:
    """A Head emits one action per value and the agent axis is the observation's, so the team's count
    changes how many actions come back and never how many the Head chooses between."""
    head = make_network(spaces(TEAM)["discrete"].observation_space, gym.spaces.Discrete(ACTIONS)).policy

    assert head.num_actions == ACTIONS


def test_one_assembly_serves_every_environment(space: Space) -> None:
    """A masked observation is the Head's business and nothing else's, so an env that has a mask and
    an env that has none are one network apart from which Head it holds."""
    assert type(make_network(space.observation_space, space.action_space)) is PPONetwork


@pytest.mark.parametrize(
    "action_space",
    [
        pytest.param(gym.spaces.Box(-1.0, 1.0, (ACTIONS, 2)), id="a Box of two dimensions"),
        pytest.param(gym.spaces.MultiDiscrete([2, ACTIONS]), id="several discrete axes"),
    ],
)
def test_an_action_space_with_no_head_is_refused_where_the_network_is_built(action_space: gym.Space) -> None:
    """Rather than at the first act, where the same mismatch is a shape error three layers down."""
    with pytest.raises(ValueError):
        make_network(gym.spaces.Box(-1.0, 1.0, (OBS,)), action_space)


@pytest.mark.parametrize(
    ("action_space", "trailing"),
    [
        pytest.param(gym.spaces.Discrete(ACTIONS), (), id="discrete"),
        pytest.param(gym.spaces.Box(-1.0, 1.0, (ACTIONS,)), (ACTIONS,), id="continuous"),
    ],
)
@pytest.mark.parametrize("num_agents", [1, TEAM], ids=["one agent", "a team"])
def test_a_trajectory_slot_takes_the_action_space_s_own_shape(action_space, trailing, num_agents: int) -> None:
    """A slot sized for a scalar takes a continuous action's first component and drops the rest. The
    space is one agent's, so the count is spelled here rather than pushed into a space a Head reads."""
    axis = prefix(BATCH, num_agents)
    slots = example(ROLLOUT, axis, PLAIN_SPACE, action_space)

    assert slots[TRANSITIONS].action.shape == (SLOTS, *axis, *trailing)
    assert slots[TRANSITIONS].action.dtype == action_space.dtype


@pytest.mark.parametrize("num_agents", [1, TEAM], ids=["one agent", "a team"])
def test_every_field_but_the_observation_leads_with_the_prefix(num_agents: int) -> None:
    """The observation keeps whatever shape the env gave it, because no stepper touches one and the
    Assembly is what reads it. Everything an Estimator or a mask reads leads with what the stepper
    reported — which has an agent axis for a team and none at all for a single agent."""
    axis = prefix(BATCH, num_agents)
    transitions = example(ROLLOUT, axis, PLAIN_SPACE, gym.spaces.Discrete(ACTIONS))[TRANSITIONS]

    width = (SLOTS, *axis)
    assert transitions.reward.shape == transitions.value.shape == transitions.log_prob.shape == width
    assert transitions.terminated.shape == transitions.truncated.shape == width
    assert transitions.observation.shape == (SLOTS, BATCH, OBS)


@pytest.mark.parametrize("num_agents", [1, TEAM], ids=["one agent", "a team"])
def test_a_mask_needs_no_slot_of_its_own(num_agents: int) -> None:
    """It rides in the observation, so the Trajectory that carries one is the Trajectory that
    already carries the other — no Transition field, no accumulator change, no Roller change."""
    masked = spaces(num_agents)["masked"]
    slots = example(ROLLOUT, prefix(BATCH, num_agents), masked.observation_space, masked.action_space)

    assert slots[TRANSITIONS].observation[ACTION_MASK].shape == (SLOTS, BATCH, *masked.agent_axis, ACTIONS)
    assert slots[TRANSITIONS].observation[OBSERVATION].shape == (SLOTS, BATCH, *masked.agent_axis, OBS)


def fresh_loss(space: Space) -> tuple[jax.Array, dict]:
    """One loss taken at initialisation, over an Assembly built for ``space`` alone rather than from
    the fixtures: what a fresh policy's entropy reads is the one thing the two do not share.
    """
    network = make_network(space.observation_space, space.action_space)
    params = initialiser(space, network)(jax.random.key(0))
    rule = rule_over(space, network)
    return rule.loss(params, PPOBatch(*from_trajectory(rule, acted(space, network, params))))


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
