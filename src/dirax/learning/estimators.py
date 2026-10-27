"""Estimators: a Trajectory's rewards and values to the advantages and targets an Objective reads.

The boundary translation ADR-0010 mandates lives here and nowhere else. Slots are T+1 and
time-major, and every derived signal reads the flags at a shift of zero per ADR-0001.
"""

from functools import partial

import jax
import jax.numpy as jnp

from dirax.types import Array


# I may use the regular scan for GAE in future, this is faster, but it's much less readable
def backward_affine_scan(increment: jax.Array, coefficient: jax.Array, axis: int = 0) -> jax.Array:
    """Solve ``y_t = increment_t + coefficient_t * y_{t+1}`` down ``axis``, from ``y_T = 0``.

    Useful for GAE and the like.

    Me: Can you express GAE as a reverse cumulative sum?
    Claude: hold my beer.

    Args:
        increment: Per-slot addend, slots down ``axis``.
        coefficient: Per-slot decay applied to the accumulation from the slot after, same shape.
            Zero severs the recurrence, which is how an episode boundary is expressed.
        axis: Which axis the slots run down. Every other axis is mapped over implicitly.

    Returns:
        The accumulation, shaped like ``increment``.
    """

    def compose(later: tuple[jax.Array, jax.Array], earlier: tuple[jax.Array, jax.Array]):
        increment_later, coefficient_later = later
        increment_earlier, coefficient_earlier = earlier
        return (
            increment_earlier + coefficient_earlier * increment_later,
            coefficient_earlier * coefficient_later,
        )

    accumulated, _ = jax.lax.associative_scan(compose, (increment, coefficient), reverse=True, axis=axis)
    return accumulated


def generalised_advantages(
    reward: Array,
    value: Array,
    terminated: Array,
    truncated: Array,
    gamma: float,
    gae_lambda: float,
    axis: int = 0,
) -> jax.Array:
    """GAE over T+1 aligned slots, reading both boundary flags apart.

    The implementation is a reverse cumulative sum and as GAE is associative, it can be done in parallel.

    Args:
        reward: ``(T+1, ...)``, what each slot's action earned.
        value: ``(T+1, ...)``, of each slot's own observation.
        terminated: ``(T+1, ...)`` bool, whether each slot's observation is a Final obs.
        truncated: ``(T+1, ...)`` bool, whether each slot's observation was cut by a time limit.
        gamma: Discount.
        gae_lambda: Bias-variance trade between the one-step residual and the full return.
        axis: Which axis the slots run down. Every other axis is mapped over implicitly, so a
            batch-major caller passes ``axis=1`` rather than wrapping this in a ``vmap``.

    Returns:
        Estimates of length T on ``axis``. The trailing slot is consumed as the bootstrap, never returned.
    """
    leaving = partial(jax.lax.slice_in_dim, start_index=0, limit_index=-1, axis=axis)
    landing = partial(jax.lax.slice_in_dim, start_index=1, limit_index=None, axis=axis)

    bootstrap = jnp.logical_not(landing(terminated)).astype(value.dtype)
    continues = bootstrap * jnp.logical_not(landing(truncated)).astype(value.dtype)

    residual = leaving(reward) + gamma * bootstrap * landing(value) - leaving(value)
    return backward_affine_scan(residual, gamma * gae_lambda * continues, axis)
