import inspect
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from dirax.networks.head import ContinuousHead, DiscreteHead, MaskedDiscreteHead, ValueHead

NUM_ENVS = 4
EMBEDDING = 8
NUM_ACTIONS = 4
LEGAL, ILLEGAL = 0, 1


def test_discrete_head_answers_the_four_calls_a_loss_makes(
    leading: tuple[int, ...], apply_module: Callable[..., Any]
) -> None:
    distribution = apply_module(DiscreteHead(num_actions=NUM_ACTIONS), jnp.ones((*leading, EMBEDDING)))

    action = distribution.sample(seed=jax.random.key(0))

    assert action.shape == leading
    assert distribution.log_prob(action).shape == leading
    assert distribution.entropy().shape == leading
    assert distribution.mode().shape == leading


def test_discrete_head_exposes_its_logits_without_sampling(
    leading: tuple[int, ...], apply_module: Callable[..., Any]
) -> None:
    """APPO's KL term stores the behaviour policy's parameters in a Trajectory, not a log-prob."""
    distribution = apply_module(DiscreteHead(num_actions=NUM_ACTIONS), jnp.ones((*leading, EMBEDDING)))

    assert distribution.logits.shape == (*leading, NUM_ACTIONS)


def test_discrete_head_starts_near_uniform_from_its_hundredth_policy_gain(
    apply_module: Callable[..., Any],
) -> None:
    """What the small gain on a policy output buys: no action favoured before training.

    distrax normalises the logits it is handed, so the kernel is not recoverable through them
    and the gain is asserted through its consequence. Root-two here would be nowhere near.
    """
    distribution = apply_module(DiscreteHead(num_actions=NUM_ACTIONS), jnp.ones((NUM_ENVS, EMBEDDING)))

    assert jnp.allclose(distribution.probs, 1.0 / NUM_ACTIONS, atol=0.015)


def test_value_head_returns_one_scalar_per_element(leading: tuple[int, ...], apply_module: Callable[..., Any]) -> None:
    """No trailing singleton axis, so no algorithm carries one by accident."""
    value = apply_module(ValueHead(), jnp.ones((*leading, EMBEDDING)))

    assert value.shape == leading


def test_value_head_initialises_orthogonally_at_unit_gain(apply_module: Callable[..., Any]) -> None:
    value = apply_module(ValueHead(), jnp.eye(EMBEDDING))

    assert jnp.allclose(jnp.linalg.norm(value), 1.0, atol=1e-6)


def action_mask(leading: tuple[int, ...]) -> jax.Array:
    """Every action legal but ``ILLEGAL``, so entropy stays finite and sampling has a choice."""
    return jnp.ones((*leading, NUM_ACTIONS), bool).at[..., ILLEGAL].set(False)


def test_masked_head_applies_at_both_layouts(leading: tuple[int, ...], apply_module: Callable[..., Any]) -> None:
    embedding = jnp.ones((*leading, EMBEDDING))

    distribution = apply_module(MaskedDiscreteHead(num_actions=NUM_ACTIONS), embedding, action_mask(leading))

    assert distribution.sample(seed=jax.random.key(0)).shape == leading
    assert distribution.logits.shape == (*leading, NUM_ACTIONS)


def test_masked_head_never_samples_a_masked_action(apply_module: Callable[..., Any]) -> None:
    leading = (NUM_ENVS,)
    distribution = apply_module(
        MaskedDiscreteHead(num_actions=NUM_ACTIONS), jnp.ones((*leading, EMBEDDING)), action_mask(leading)
    )

    keys = jax.random.split(jax.random.key(0), 2000)
    actions = jax.vmap(lambda key: distribution.sample(seed=key))(keys)

    assert not (actions == ILLEGAL).any()


def test_a_masked_action_has_a_log_probability_of_negative_infinity(apply_module: Callable[..., Any]) -> None:
    """So it contributes nothing to any gradient."""
    leading = (NUM_ENVS,)
    distribution = apply_module(
        MaskedDiscreteHead(num_actions=NUM_ACTIONS), jnp.ones((*leading, EMBEDDING)), action_mask(leading)
    )

    masked = distribution.log_prob(jnp.full(leading, ILLEGAL))
    legal = distribution.log_prob(jnp.full(leading, LEGAL))

    assert jnp.isneginf(masked).all()
    assert jnp.isfinite(legal).all()


def test_masked_head_entropy_is_finite_while_one_action_is_legal(apply_module: Callable[..., Any]) -> None:
    leading = (NUM_ENVS,)
    one_legal = jnp.zeros((*leading, NUM_ACTIONS), bool).at[..., LEGAL].set(True)

    for mask in (action_mask(leading), one_legal):
        distribution = apply_module(MaskedDiscreteHead(num_actions=NUM_ACTIONS), jnp.ones((*leading, EMBEDDING)), mask)
        assert jnp.isfinite(distribution.entropy()).all()


def test_the_unmasked_discrete_head_takes_no_mask() -> None:
    """Masking is a distinct type, so neither Head carries the other's parameter."""
    assert list(inspect.signature(DiscreteHead.__call__).parameters) == ["self", "embedding"]
    assert list(inspect.signature(MaskedDiscreteHead.__call__).parameters) == ["self", "embedding", "action_mask"]


def test_continuous_head_applies_at_both_layouts(leading: tuple[int, ...], apply_module: Callable[..., Any]) -> None:
    distribution = apply_module(ContinuousHead(num_actions=NUM_ACTIONS), jnp.ones((*leading, EMBEDDING)))

    action = distribution.sample(seed=jax.random.key(0))

    assert action.shape == (*leading, NUM_ACTIONS)
    assert distribution.log_prob(action).shape == leading
    assert distribution.entropy().shape == leading


def test_continuous_head_exposes_the_base_location_and_scale(
    leading: tuple[int, ...], apply_module: Callable[..., Any]
) -> None:
    distribution = apply_module(ContinuousHead(num_actions=NUM_ACTIONS), jnp.ones((*leading, EMBEDDING)))

    assert distribution.loc.shape == (*leading, NUM_ACTIONS)
    assert distribution.scale.shape == (*leading, NUM_ACTIONS)


def test_continuous_head_samples_strictly_inside_the_interval(apply_module: Callable[..., Any]) -> None:
    distribution = apply_module(ContinuousHead(num_actions=NUM_ACTIONS), jnp.ones((NUM_ENVS, EMBEDDING)))

    action = distribution.sample(seed=jax.random.key(0), sample_shape=1000)

    assert (jnp.abs(action) < 1.0).all()


def test_continuous_head_keeps_the_scale_positive_for_extreme_embeddings(
    apply_module: Callable[..., Any],
) -> None:
    """A softplus alone rounds to zero out here, which would make the Gaussian degenerate."""
    extreme = jnp.array([-1e6, 1e6, -1e12, 1e12])[:, None] * jnp.ones((1, EMBEDDING))

    distribution = apply_module(ContinuousHead(num_actions=NUM_ACTIONS), extreme)

    assert (distribution.scale > 0.0).all()
    assert jnp.isfinite(distribution.scale).all()


def test_continuous_head_takes_no_action_mask() -> None:
    """Mava's and Stoix's accept one to match the discrete signature, then discard it."""
    assert list(inspect.signature(ContinuousHead.__call__).parameters) == ["self", "embedding"]


def test_continuous_head_initialises_orthogonally_at_a_hundredth(apply_module: Callable[..., Any]) -> None:
    """The location is reachable unnormalised, so applying to the identity returns the kernel."""
    kernel = apply_module(ContinuousHead(num_actions=NUM_ACTIONS), jnp.eye(EMBEDDING)).loc

    assert jnp.allclose(jnp.linalg.norm(kernel, axis=0), 0.01, atol=1e-6)
