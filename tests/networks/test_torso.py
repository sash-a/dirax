from collections.abc import Callable
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from brrrl.networks.torso import ImpalaCNNTorso, MLPTorso, NatureCNNTorso

NUM_LANES = 4
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

    embedding = apply_module(torso, jnp.ones((NUM_LANES, OBS_FEATURES)))

    assert (embedding == 3.0).all()


def test_mlp_normalises_each_embedding_only_when_layer_norm_is_enabled(apply_module: Callable[..., Any]) -> None:
    observation = 5.0 * jax.random.normal(jax.random.key(1), (NUM_LANES, OBS_FEATURES))

    normalised = apply_module(MLPTorso(layer_sizes=(16,), activation=identity, layer_norm=True), observation)
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
    flat = apply_module(torso.build(activation=lambda x: jnp.full_like(x, 3.0)), frames((NUM_LANES,), torso.pixels))

    assert (flat == 3.0).all()


@pytest.mark.parametrize("torso", CONV_TORSOS)
def test_conv_torso_layer_norm_takes_the_input_scale_out_of_the_embedding(
    torso: ConvTorso, apply_module: Callable[..., Any]
) -> None:
    loud = frames((NUM_LANES,), torso.pixels, scale=100.0)

    normalised = apply_module(torso.build(layer_norm=True), loud)
    plain = apply_module(torso.build(), loud)

    assert jnp.abs(normalised).max() < jnp.abs(plain).max() / 10.0
