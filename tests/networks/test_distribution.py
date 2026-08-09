"""Every property here is a regression test for something observed to break during design.

A subtly wrong log-probability trains fine while learning worse, so a dependency upgrade that
reintroduces one of these has to fail loudly rather than quietly.
"""

import inspect

import distrax
import jax
import jax.numpy as jnp
import pytest

from brrrl.networks.distribution import TanhNormal

NUM_ACTIONS = 3
MONTE_CARLO_SAMPLES = 200_000


def one_action(scale: float) -> TanhNormal:
    return TanhNormal(jnp.zeros(1), jnp.full((1,), scale))


def log_prob_of(loc: jax.Array, scale: jax.Array, action: jax.Array) -> jax.Array:
    return TanhNormal(loc, scale).log_prob(action)


@pytest.mark.parametrize("action", [-1.0, 1.0, -1.5, 2.0])
def test_log_prob_is_finite_at_and_beyond_the_action_boundary(action: float) -> None:
    """Bare distrax returns NaN here, because the inverse of the bijector diverges."""
    log_prob = one_action(1.0).log_prob(jnp.full((1,), action))

    assert jnp.isfinite(log_prob).all()


@pytest.mark.parametrize("action", [-1.0, 1.0])
def test_log_prob_gradient_is_finite_at_the_action_boundary(action: float) -> None:
    """The hazard that silently destroys a run: one saturated action, every param NaN."""
    gradients = jax.grad(log_prob_of, argnums=(0, 1, 2))(jnp.zeros(1), jnp.ones(1), jnp.full((1,), action))

    assert all(jnp.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_entropy_agrees_with_a_high_sample_monte_carlo_reference(scale: float) -> None:
    """The reference goes nowhere near our own ``log_prob``, or a jointly wrong pair would pass.

    It is the same identity — base entropy plus the expected jacobian correction — estimated by
    sampling distrax directly instead of by quadrature, which is the claim under test.
    """
    base = distrax.Normal(jnp.zeros(1), jnp.full((1,), scale))
    draws = base.sample(seed=jax.random.key(0), sample_shape=MONTE_CARLO_SAMPLES)
    correction = distrax.Tanh().forward_log_det_jacobian(draws).mean()

    reference = base.entropy().sum() + correction

    assert jnp.allclose(one_action(scale).entropy(), reference, atol=0.02)


@pytest.mark.parametrize("scale", [0.3, 0.5])
def test_entropy_rises_with_scale_below_its_peak(scale: float) -> None:
    assert jax.grad(lambda s: one_action(s).entropy())(scale) > 0.0


@pytest.mark.parametrize("scale", [0.9, 1.0, 1.5, 2.0, 3.0])
def test_entropy_falls_with_scale_above_its_peak(scale: float) -> None:
    """The mode-based estimator this replaces has the derivative negative nowhere, so an
    entropy bonus built on it rewards widening the policy without limit."""
    assert jax.grad(lambda s: one_action(s).entropy())(scale) < 0.0


def test_entropy_takes_no_key_on_either_distribution() -> None:
    """So a loss calls the same four methods with no branch on the action space."""
    continuous = TanhNormal(jnp.zeros(NUM_ACTIONS), jnp.ones(NUM_ACTIONS))
    discrete = distrax.Categorical(logits=jnp.zeros(NUM_ACTIONS))

    for distribution in (continuous, discrete):
        assert jnp.isfinite(distribution.entropy()).all()
        assert list(inspect.signature(distribution.entropy).parameters) == []


def test_log_prob_is_the_sum_over_the_action_axis() -> None:
    """Summed here rather than by nesting in distrax's independent wrapper, whose entropy
    drops the keyword arguments tfp's forwards, so nesting raises on the seed."""
    loc, scale = jnp.array([0.1, -0.2, 0.3]), jnp.array([0.5, 1.0, 2.0])
    action = jnp.array([0.4, -0.6, 0.2])

    joint = TanhNormal(loc, scale).log_prob(action)
    separately = sum(
        TanhNormal(loc[i : i + 1], scale[i : i + 1]).log_prob(action[i : i + 1]) for i in range(NUM_ACTIONS)
    )

    assert jnp.allclose(joint, separately, atol=1e-5)


def test_the_four_calls_return_the_expected_shapes_at_both_layouts(leading: tuple[int, ...]) -> None:
    distribution = TanhNormal(jnp.zeros((*leading, NUM_ACTIONS)), jnp.ones((*leading, NUM_ACTIONS)))

    action = distribution.sample(seed=jax.random.key(0))

    assert action.shape == (*leading, NUM_ACTIONS)
    assert distribution.mode().shape == (*leading, NUM_ACTIONS)
    assert distribution.log_prob(action).shape == leading
    assert distribution.entropy().shape == leading


def test_sampled_actions_stay_inside_the_interval() -> None:
    """No rescaling onto environment bounds happens here; that is the wrappers package."""
    distribution = TanhNormal(jnp.zeros(NUM_ACTIONS), jnp.ones(NUM_ACTIONS))

    action = distribution.sample(seed=jax.random.key(0), sample_shape=1000)

    assert (jnp.abs(action) < 1.0).all()


def test_the_mode_is_the_base_mode_squashed() -> None:
    loc = jnp.array([0.5, -1.0, 2.0])

    mode = TanhNormal(loc, jnp.ones(NUM_ACTIONS)).mode()

    assert jnp.allclose(mode, jnp.tanh(loc))


def test_the_base_location_and_scale_stay_reachable() -> None:
    """APPO's KL term needs the behaviour policy's parameters in a Trajectory."""
    loc, scale = jnp.array([0.1, -0.2, 0.3]), jnp.array([0.5, 1.0, 2.0])

    distribution = TanhNormal(loc, scale)

    assert jnp.array_equal(distribution.loc, loc)
    assert jnp.array_equal(distribution.scale, scale)


def test_the_distribution_survives_jit() -> None:
    """A Learner carries this through a scanned pure function, so it has to trace."""

    @jax.jit
    def entropy(loc: jax.Array, scale: jax.Array) -> jax.Array:
        return TanhNormal(loc, scale).entropy()

    assert jnp.isfinite(entropy(jnp.zeros(NUM_ACTIONS), jnp.ones(NUM_ACTIONS))).all()
