from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from brrrl.networks.torso import ImpalaCNNTorso, MLPTorso, NatureCNNTorso

NUM_ENVS = 4
OBS_FEATURES = 8
EMBEDDING = 8


def identity(x: jax.Array) -> jax.Array:
    return x


def test_mlp_embeds_at_both_layouts(leading: tuple[int, ...], apply_module: Callable[..., Any]) -> None:
    embedding = apply_module(MLPTorso(layer_sizes=(16, 6)), jnp.ones((*leading, OBS_FEATURES)))

    assert embedding.shape == (*leading, 6)


def test_mlp_accepts_an_arbitrary_callable_as_its_activation(apply_module: Callable[..., Any]) -> None:
    """Not a string looked up in a table, so no activation is out of reach."""
    torso = MLPTorso(layer_sizes=(6,), activation=lambda x: jnp.full_like(x, 3.0))

    embedding = apply_module(torso, jnp.ones((NUM_ENVS, OBS_FEATURES)))

    assert (embedding == 3.0).all()


def test_mlp_normalises_each_embedding_only_when_given_a_normalisation(apply_module: Callable[..., Any]) -> None:
    observation = 5.0 * jax.random.normal(jax.random.key(1), (NUM_ENVS, OBS_FEATURES))

    normalised = apply_module(MLPTorso(layer_sizes=(16,), activation=identity, normalisation=nn.LayerNorm), observation)
    plain = apply_module(MLPTorso(layer_sizes=(16,), activation=identity), observation)

    assert jnp.allclose(normalised.mean(axis=-1), 0.0, atol=1e-5)
    assert jnp.allclose(normalised.std(axis=-1), 1.0, atol=1e-3)
    assert (plain.std(axis=-1) > 2.0).all()


def test_mlp_initialises_orthogonally_at_root_two_gain(apply_module: Callable[..., Any]) -> None:
    """Applying to the identity returns the kernel, whose columns are orthogonal with norm ``gain``."""
    torso = MLPTorso(layer_sizes=(EMBEDDING,), activation=identity)

    kernel = apply_module(torso, jnp.eye(OBS_FEATURES))

    assert jnp.allclose(jnp.linalg.norm(kernel, axis=0), jnp.sqrt(2.0), atol=1e-5)


CHANNELS = 3
NATURE_FRAME, NATURE_EMBEDDING = 84, 7 * 7 * 64
IMPALA_FRAME, IMPALA_EMBEDDING = 64, 8 * 8 * 32


class ConvTorso(NamedTuple):
    """One convolutional Torso, with the frame width it takes and the embedding it publishes."""

    build: type[nn.Module]
    pixels: int
    embedding: int


CONV_TORSOS = [
    pytest.param(ConvTorso(NatureCNNTorso, NATURE_FRAME, NATURE_EMBEDDING), id="nature"),
    pytest.param(ConvTorso(ImpalaCNNTorso, IMPALA_FRAME, IMPALA_EMBEDDING), id="impala"),
]


def frames(leading: tuple[int, ...], pixels: int, scale: float = 1.0) -> jax.Array:
    return scale * jax.random.normal(jax.random.key(1), (*leading, pixels, pixels, CHANNELS))


@pytest.mark.parametrize("torso", CONV_TORSOS)
def test_conv_torso_flattens_to_the_published_embedding_at_both_layouts(
    torso: ConvTorso, leading: tuple[int, ...], apply_module: Callable[..., Any]
) -> None:
    """The embedding width pins the channel counts, kernel sizes and strides in one number."""
    flat = apply_module(torso.build(), frames(leading, torso.pixels))

    assert flat.shape == (*leading, torso.embedding)


@pytest.mark.parametrize("torso", CONV_TORSOS)
def test_conv_torso_accepts_an_arbitrary_callable_as_its_activation(
    torso: ConvTorso, apply_module: Callable[..., Any]
) -> None:
    flat = apply_module(torso.build(activation=lambda x: jnp.full_like(x, 3.0)), frames((NUM_ENVS,), torso.pixels))

    assert (flat == 3.0).all()


@pytest.mark.parametrize("torso", CONV_TORSOS)
def test_conv_torso_normalisation_takes_the_input_scale_out_of_the_embedding(
    torso: ConvTorso, apply_module: Callable[..., Any]
) -> None:
    loud = frames((NUM_ENVS,), torso.pixels, scale=100.0)

    normalised = apply_module(torso.build(normalisation=nn.LayerNorm), loud)
    plain = apply_module(torso.build(), loud)

    assert jnp.abs(normalised).max() < jnp.abs(plain).max() / 10.0


@pytest.mark.parametrize(
    "normalisation",
    [nn.LayerNorm, nn.RMSNorm, partial(nn.GroupNorm, num_groups=4)],
    ids=["layer", "rms", "group"],
)
def test_any_stateless_normalisation_can_be_passed_in(
    normalisation: Callable[..., nn.Module], apply_module: Callable[..., Any]
) -> None:
    torso = MLPTorso(layer_sizes=(16,), activation=identity, normalisation=normalisation)

    embedding = apply_module(torso, 5.0 * jax.random.normal(jax.random.key(1), (NUM_ENVS, OBS_FEATURES)))

    assert embedding.shape == (NUM_ENVS, 16)
    assert (jnp.abs(embedding) < 5.0).all()


def test_a_stateful_normalisation_can_be_passed_in_bound_up_front() -> None:
    """Batch statistics land in their own collection, which is the caller's to thread."""
    torso = MLPTorso(layer_sizes=(16,), normalisation=partial(nn.BatchNorm, use_running_average=False))
    observation = jax.random.normal(jax.random.key(1), (NUM_ENVS, OBS_FEATURES))

    variables = torso.init(jax.random.key(0), observation)
    embedding, updates = torso.apply(variables, observation, mutable=["batch_stats"])

    assert embedding.shape == (NUM_ENVS, 16)
    assert "batch_stats" in variables
    assert "batch_stats" in updates


def test_a_different_kernel_init_is_honoured(apply_module: Callable[..., Any]) -> None:
    """No gain follows from a callable activation, so tanh's 5/3 has to be passable."""
    torso = MLPTorso(layer_sizes=(EMBEDDING,), activation=identity, kernel_init=nn.initializers.orthogonal(5 / 3))

    kernel = apply_module(torso, jnp.eye(OBS_FEATURES))

    assert jnp.allclose(jnp.linalg.norm(kernel, axis=0), 5 / 3, atol=1e-5)
