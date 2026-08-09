import math
from functools import lru_cache
from typing import final

import distrax
import jax
import jax.numpy as jnp
import numpy as np

THRESHOLD = 0.999
QUADRATURE_NODES = 9


@lru_cache
def gauss_hermite(num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights approximating ``integral of exp(-z^2) f(z) dz`` over the real line."""
    return np.polynomial.hermite.hermgauss(num_nodes)


@final
class TanhNormal(distrax.Transformed):
    """A Gaussian squashed into ``(-1, 1)``, answering the four calls a loss makes.

    No library ships this. distrax, numpyro and distreqx all raise on the entropy of a
    tanh-transformed Gaussian, and bare distrax returns NaN for both the log-probability and
    its gradient at the boundary of the action range.

    Actions land in ``(-1, 1)``. Mapping them onto an environment's real bounds belongs to
    the wrappers package.

    Attributes:
        loc: Location of the base Gaussian, reachable so a Trajectory can carry it.
        scale: Scale of the base Gaussian, reachable for the same reason.
    """

    def __init__(
        self,
        loc: jax.Array,
        scale: jax.Array,
        threshold: float = THRESHOLD,
        num_quadrature_nodes: int = QUADRATURE_NODES,
    ) -> None:
        """Precompute the log-probability mass beyond each side of ``threshold``.

        Args:
            loc: Location of the base Gaussian, with the action space as its trailing axis.
            scale: Scale of the base Gaussian, shaped like ``loc``.
            threshold: Where ``log_prob`` switches from the transformed density to that
                interval's average density. Keeps it finite at the boundary.
            num_quadrature_nodes: Nodes integrating the entropy's jacobian correction. The
                estimate degrades above a scale of roughly four; this is the knob for that.
        """
        loc, scale = jnp.asarray(loc), jnp.asarray(scale)
        super().__init__(distrax.Normal(loc=loc, scale=scale), distrax.Tanh())
        self._threshold = threshold
        self._num_quadrature_nodes = num_quadrature_nodes

        inverse_threshold = self.bijector.inverse(threshold)
        log_width = math.log(1.0 - threshold)
        # Both tail masses go through log_ndtr, which has no half-precision implementation.
        tails = distrax.Normal(loc=loc.astype(jnp.float32), scale=scale.astype(jnp.float32))
        self._log_prob_left = (tails.log_cdf(-inverse_threshold) - log_width).astype(loc.dtype)
        self._log_prob_right = (tails.log_survival_function(inverse_threshold) - log_width).astype(loc.dtype)

    @property
    def loc(self) -> jax.Array:
        return self.distribution.loc

    @property
    def scale(self) -> jax.Array:
        return self.distribution.scale

    def log_prob(self, value: jax.Array) -> jax.Array:
        """Log-probability of ``value``, summed over the action axis.

        Outside the threshold the transformed density gives way to that interval's average
        density, which is what keeps this and its gradient finite at an action of exactly
        ``+/- 1``. Summed here rather than by nesting in ``distrax.Independent``, whose
        ``entropy`` drops the keyword arguments tfp's forwards.
        """
        clipped = jnp.clip(value, -self._threshold, self._threshold)
        log_prob = jnp.where(
            value <= -self._threshold,
            self._log_prob_left,
            jnp.where(value >= self._threshold, self._log_prob_right, super().log_prob(clipped)),
        )
        return jnp.sum(log_prob, axis=-1)

    def mode(self) -> jax.Array:
        return self.bijector.forward(self.distribution.mode())

    def entropy(self) -> jax.Array:  # ty: ignore[invalid-method-override]
        """Base entropy plus its jacobian correction, by Gauss-Hermite quadrature.

        Summed over the action axis, and takes no argument at all — not even the base
        signature's ``input_hint`` — so a loss calls this and a categorical's ``entropy``
        identically. ADR-0007 records why quadrature rather than sampling or the mode.
        """
        nodes, weights = gauss_hermite(self._num_quadrature_nodes)
        draws = self.loc[..., None] + math.sqrt(2.0) * self.scale[..., None] * nodes
        correction = jnp.sum(weights * self.bijector.forward_log_det_jacobian(draws), axis=-1) / math.sqrt(math.pi)
        return jnp.sum(self.distribution.entropy() + correction, axis=-1)
