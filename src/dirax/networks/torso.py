# A few simple network building blocks for the most common environments
import math
from collections.abc import Callable, Sequence
from typing import final

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.typing import Dtype, Initializer

type Activation = Callable[[jax.Array], jax.Array]
type Normalisation = Callable[..., Activation]

TORSO_GAIN = math.sqrt(2.0)
TORSO_KERNEL_INIT = nn.initializers.orthogonal(TORSO_GAIN)

KERNEL = (3, 3)
STRIDE = (1, 1)
POOL_WINDOW = (3, 3)
POOL_STRIDES = (2, 2)

# Channels, kernel and strides per layer of the published Nature-DQN architecture.
NATURE_STACK = ((32, (8, 8), (4, 4)), (64, (4, 4), (2, 2)), (64, KERNEL, STRIDE))
IMPALA_CHANNELS = (16, 32, 32)
IMPALA_RESIDUAL_UNITS = 2


def no_normalisation(**dtypes: Dtype | None) -> Activation:
    """The do-nothing normalisation, so that no Torso carries a branch for the common case."""
    return lambda embedding: embedding


@final
class MLPTorso(nn.Module):
    """Dense stack mapping an observation to an embedding.

    The activation follows every layer, including the last, so the embedding is what a Head
    consumes rather than a raw pre-activation.

    Attributes:
        layer_sizes: Width of each Dense layer, in order.
        activation: Applied after every layer.
        normalisation: Builds the layer normalising each output before it is activated, so any
            Flax norm fits: ``nn.LayerNorm``, ``nn.RMSNorm``, ``partial(nn.GroupNorm, ...)``.
            It builds rather than is one so that each layer gets its own params.
        kernel_init: Root-two gain suits the default ReLU. Tanh wants 5/3 so it is configurable.
        param_dtype: Dtype params are stored in.
        dtype: Dtype the layers compute in, promoted from the inputs and params when None.
    """

    layer_sizes: Sequence[int]
    activation: Activation = nn.relu
    normalisation: Normalisation = no_normalisation
    kernel_init: Initializer = TORSO_KERNEL_INIT
    param_dtype: Dtype = jnp.float32
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        embedding = observation
        for size in self.layer_sizes:
            embedding = nn.Dense(
                size,
                kernel_init=self.kernel_init,
                param_dtype=self.param_dtype,
                dtype=self.dtype,
            )(embedding)
            embedding = self.normalisation(param_dtype=self.param_dtype, dtype=self.dtype)(embedding)
            embedding = self.activation(embedding)
        return embedding


def flatten(embedding: jax.Array) -> jax.Array:
    """Collapse an image's height, width and channel axes into one embedding axis.

    The only shape collapse in the package. It leaves the leading axes untouched, so acting at
    ``(E, ...)`` and learning at ``(T, E, ...)`` still call the identical module.
    """
    return embedding.reshape(*embedding.shape[:-3], -1)


@final
class CNNTorso(nn.Module):
    """The Nature-DQN convolutional stack, mapping a pixel observation to a flat embedding.

    Attributes:
        activation: Applied after every convolution.
        normalisation: Builds the layer normalising each convolution's output, as on ``MLPTorso``.
        kernel_init: As on ``MLPTorso`` — pick one that suits ``activation``.
        param_dtype: Dtype params are stored in. Orthogonal initialisation runs a QR, which
            has no half-precision kernel, so reach for ``dtype`` rather than this one.
        dtype: Dtype the layers compute in, promoted from the inputs and params when None.
    """

    activation: Activation = nn.relu
    normalisation: Normalisation = no_normalisation
    kernel_init: Initializer = TORSO_KERNEL_INIT
    param_dtype: Dtype = jnp.float32
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        embedding = observation
        for channels, kernel, strides in NATURE_STACK:
            embedding = nn.Conv(
                channels,
                kernel,
                strides,
                padding="VALID",
                kernel_init=self.kernel_init,
                param_dtype=self.param_dtype,
                dtype=self.dtype,
            )(embedding)
            embedding = self.normalisation(param_dtype=self.param_dtype, dtype=self.dtype)(embedding)
            embedding = self.activation(embedding)
        return flatten(embedding)


@final
class ResidualCNNTorso(nn.Module):
    """The IMPALA residual convolutional stack, mapping a pixel observation to a flat embedding.

    Each block is a convolution and a max-pool followed by ``num_residual_units`` residual
    units, every convolution three by three at unit stride.

    Attributes:
        channel_sizes: Width of each block, in order. The published deep architecture by default.
        num_residual_units: Residual units following each block's max-pool.
        activation: Applied before every convolution in a residual unit, and once at the end.
        normalisation: Builds the layer normalising every convolution's output, as on ``MLPTorso``.
        kernel_init: As on ``MLPTorso`` — pick one that suits ``activation``.
        param_dtype: Dtype params are stored in. Orthogonal initialisation runs a QR, which
            has no half-precision kernel, so reach for ``dtype`` rather than this one.
        dtype: Dtype the layers compute in, promoted from the inputs and params when None.
    """

    channel_sizes: Sequence[int] = IMPALA_CHANNELS
    num_residual_units: int = IMPALA_RESIDUAL_UNITS
    activation: Activation = nn.relu
    normalisation: Normalisation = no_normalisation
    kernel_init: Initializer = TORSO_KERNEL_INIT
    param_dtype: Dtype = jnp.float32
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        embedding = observation
        for channels in self.channel_sizes:
            embedding = self.convolve(channels, embedding)
            embedding = nn.max_pool(embedding, POOL_WINDOW, POOL_STRIDES, padding="SAME")
            for _ in range(self.num_residual_units):
                embedding = embedding + self.residual_unit(channels, embedding)
        return flatten(self.activation(embedding))

    def convolve(self, channels: int, embedding: jax.Array) -> jax.Array:
        convolved = nn.Conv(
            channels,
            KERNEL,
            STRIDE,
            padding="SAME",
            kernel_init=self.kernel_init,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )(embedding)
        return self.normalisation(param_dtype=self.param_dtype, dtype=self.dtype)(convolved)

    def residual_unit(self, channels: int, embedding: jax.Array) -> jax.Array:
        convolved = self.convolve(channels, self.activation(embedding))
        return self.convolve(channels, self.activation(convolved))
