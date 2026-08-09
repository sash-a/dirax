import math
from collections.abc import Callable, Sequence
from typing import final

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.typing import Dtype

type Activation = Callable[[jax.Array], jax.Array]
type Normalisation = Callable[..., nn.Module]

TORSO_GAIN = math.sqrt(2.0)

KERNEL = (3, 3)
STRIDE = (1, 1)
POOL_WINDOW = (3, 3)
POOL_STRIDES = (2, 2)

# Channels, kernel and strides per layer of the published Nature-DQN architecture.
NATURE_STACK = ((32, (8, 8), (4, 4)), (64, (4, 4), (2, 2)), (64, KERNEL, STRIDE))
IMPALA_CHANNELS = (16, 32, 32)
IMPALA_RESIDUAL_UNITS = 2


def normalise(
    embedding: jax.Array, normalisation: Normalisation | None, param_dtype: Dtype, dtype: Dtype | None
) -> jax.Array:
    """Normalise at the Torso's dtypes, or pass the embedding straight through when there is none.

    ``normalisation`` builds the layer rather than being one, so any Flax norm fits —
    ``nn.LayerNorm``, ``nn.RMSNorm``, ``nn.GroupNorm``, or a stateful one bound up front as
    ``partial(nn.BatchNorm, use_running_average=False)``.
    """
    if normalisation is None:
        return embedding
    return normalisation(param_dtype=param_dtype, dtype=dtype)(embedding)


@final
class MLPTorso(nn.Module):
    """Dense stack mapping an observation to an embedding.

    The activation follows every layer, including the last, so the embedding is what a Head
    consumes rather than a raw pre-activation.

    Attributes:
        layer_sizes: Width of each Dense layer, in order.
        activation: Applied after every layer.
        normalisation: Applied to each layer's output before activating it, or None for none.
        param_dtype: Dtype params are stored in. Orthogonal initialisation runs a QR, which
            has no half-precision kernel, so reach for ``dtype`` rather than this one.
        dtype: Dtype the layers compute in, promoted from the inputs and params when None.
    """

    layer_sizes: Sequence[int]
    activation: Activation = nn.relu
    normalisation: Normalisation | None = None
    param_dtype: Dtype = jnp.float32
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        embedding = observation
        for size in self.layer_sizes:
            embedding = nn.Dense(
                size,
                kernel_init=nn.initializers.orthogonal(TORSO_GAIN),
                param_dtype=self.param_dtype,
                dtype=self.dtype,
            )(embedding)
            embedding = normalise(embedding, self.normalisation, self.param_dtype, self.dtype)
            embedding = self.activation(embedding)
        return embedding


def convolution(
    channels: int,
    kernel: tuple[int, int],
    strides: tuple[int, int],
    padding: str,
    param_dtype: Dtype,
    dtype: Dtype | None,
) -> nn.Conv:
    """A Torso's convolution, orthogonally initialised at root-two gain."""
    return nn.Conv(
        channels,
        kernel,
        strides,
        padding=padding,
        kernel_init=nn.initializers.orthogonal(TORSO_GAIN),
        param_dtype=param_dtype,
        dtype=dtype,
    )


def flatten(embedding: jax.Array) -> jax.Array:
    """Collapse an image's height, width and channel axes into one embedding axis.

    The only shape collapse in the package. It leaves the leading axes untouched, so acting at
    ``(Lane, ...)`` and learning at ``(time, Lane, ...)`` still call the identical module.
    """
    return embedding.reshape(*embedding.shape[:-3], -1)


@final
class NatureCNNTorso(nn.Module):
    """The Nature-DQN convolutional stack, mapping a pixel observation to a flat embedding.

    Channel counts, kernel sizes and strides are the published ones and are not knobs: a
    stack with different ones is not this architecture. It ends at the flatten, so the
    published network's 512-unit fully-connected layer is an ``MLPTorso`` composed after it.

    Attributes:
        activation: Applied after every convolution.
        normalisation: Applied to each convolution's output before activating it, or None for none.
        param_dtype: Dtype params are stored in. Orthogonal initialisation runs a QR, which
            has no half-precision kernel, so reach for ``dtype`` rather than this one.
        dtype: Dtype the layers compute in, promoted from the inputs and params when None.
    """

    activation: Activation = nn.relu
    normalisation: Normalisation | None = None
    param_dtype: Dtype = jnp.float32
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        embedding = observation
        for channels, kernel, strides in NATURE_STACK:
            embedding = convolution(channels, kernel, strides, "VALID", self.param_dtype, self.dtype)(embedding)
            embedding = normalise(embedding, self.normalisation, self.param_dtype, self.dtype)
            embedding = self.activation(embedding)
        return flatten(embedding)


@final
class ImpalaCNNTorso(nn.Module):
    """The IMPALA residual convolutional stack, mapping a pixel observation to a flat embedding.

    Each block is a convolution and a max-pool followed by ``num_residual_units`` residual
    units, every convolution three by three at unit stride. It ends at the same ``flatten``
    the Nature stack uses, so the published network's 256-unit fully-connected layer is an
    ``MLPTorso`` composed after it.

    Attributes:
        channel_sizes: Width of each block, in order. The published deep architecture by default.
        num_residual_units: Residual units following each block's max-pool.
        activation: Applied before every convolution in a residual unit, and once at the end.
        normalisation: Applied to every convolution's output, or None for none.
        param_dtype: Dtype params are stored in. Orthogonal initialisation runs a QR, which
            has no half-precision kernel, so reach for ``dtype`` rather than this one.
        dtype: Dtype the layers compute in, promoted from the inputs and params when None.
    """

    channel_sizes: Sequence[int] = IMPALA_CHANNELS
    num_residual_units: int = IMPALA_RESIDUAL_UNITS
    activation: Activation = nn.relu
    normalisation: Normalisation | None = None
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
        convolved = convolution(channels, KERNEL, STRIDE, "SAME", self.param_dtype, self.dtype)(embedding)
        return normalise(convolved, self.normalisation, self.param_dtype, self.dtype)

    def residual_unit(self, channels: int, embedding: jax.Array) -> jax.Array:
        convolved = self.convolve(channels, self.activation(embedding))
        return self.convolve(channels, self.activation(convolved))
