from typing import final

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.typing import Dtype

from brrrl.networks.distribution import TanhNormal

POLICY_GAIN = 0.01
VALUE_GAIN = 1.0
MIN_SCALE = 1e-3


def dense(features: int, gain: float, param_dtype: Dtype, dtype: Dtype | None) -> nn.Dense:
    """A Head's final Dense, orthogonally initialised at ``gain``."""
    return nn.Dense(features, kernel_init=nn.initializers.orthogonal(gain), param_dtype=param_dtype, dtype=dtype)


@final
class DiscreteHead(nn.Module):
    """Embedding to a categorical distribution over actions.

    The logits stay reachable on the returned distribution, which APPO's KL term against the
    behaviour policy needs in a Trajectory.

    Attributes:
        num_actions: Size of the action space.
        param_dtype: Dtype params are stored in. Orthogonal initialisation runs a QR, which
            has no half-precision kernel, so reach for ``dtype`` rather than this one.
        dtype: Dtype the layer computes in, promoted from the inputs and params when None.
    """

    num_actions: int
    param_dtype: Dtype = jnp.float32
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, embedding: jax.Array) -> distrax.Categorical:
        logits = dense(self.num_actions, POLICY_GAIN, self.param_dtype, self.dtype)(embedding)
        return distrax.Categorical(logits=logits)


@final
class MaskedDiscreteHead(nn.Module):
    """Embedding and an action mask to a categorical distribution over the legal actions.

    A separate type rather than a flag on ``DiscreteHead``, so that no Head accepts a
    parameter it immediately discards. An illegal action's log-probability is negative
    infinity, so it can neither be sampled nor contribute to any gradient, while the entropy
    stays finite as long as one action is legal.

    Attributes:
        num_actions: Size of the action space.
        param_dtype: Dtype params are stored in. Orthogonal initialisation runs a QR, which
            has no half-precision kernel, so reach for ``dtype`` rather than this one.
        dtype: Dtype the layer computes in, promoted from the inputs and params when None.
    """

    num_actions: int
    param_dtype: Dtype = jnp.float32
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, embedding: jax.Array, action_mask: jax.Array) -> distrax.Categorical:
        logits = dense(self.num_actions, POLICY_GAIN, self.param_dtype, self.dtype)(embedding)
        return distrax.Categorical(logits=jnp.where(action_mask, logits, -jnp.inf))


@final
class ContinuousHead(nn.Module):
    """Embedding to a squashed Gaussian over actions in the interval ``(-1, 1)``.

    The scale is state-dependent: its own Dense through a softplus, floored at
    ``min_scale`` so that it stays positive for extreme embeddings, where a softplus alone
    rounds to zero.

    Takes no action mask, which only a central Assembly calling every Head uniformly would
    force on it (ADR-0008).

    Attributes:
        num_actions: Size of the action space.
        min_scale: Floor on the base Gaussian's scale.
        param_dtype: Dtype params are stored in. Orthogonal initialisation runs a QR, which
            has no half-precision kernel, so reach for ``dtype`` rather than this one.
        dtype: Dtype the layers compute in, promoted from the inputs and params when None.
    """

    num_actions: int
    min_scale: float = MIN_SCALE
    param_dtype: Dtype = jnp.float32
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, embedding: jax.Array) -> TanhNormal:
        loc = dense(self.num_actions, POLICY_GAIN, self.param_dtype, self.dtype)(embedding)
        scale = dense(self.num_actions, POLICY_GAIN, self.param_dtype, self.dtype)(embedding)
        return TanhNormal(loc, jax.nn.softplus(scale) + self.min_scale)


@final
class ValueHead(nn.Module):
    """Embedding to one scalar per element, with no trailing singleton axis.

    Attributes:
        param_dtype: Dtype params are stored in. Orthogonal initialisation runs a QR, which
            has no half-precision kernel, so reach for ``dtype`` rather than this one.
        dtype: Dtype the layer computes in, promoted from the inputs and params when None.
    """

    param_dtype: Dtype = jnp.float32
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, embedding: jax.Array) -> jax.Array:
        value = dense(1, VALUE_GAIN, self.param_dtype, self.dtype)(embedding)
        return jnp.squeeze(value, axis=-1)
