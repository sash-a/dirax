import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.typing import Dtype, Initializer, Shape


def orthogonal(gain: float) -> Initializer:
    """Orthogonal initialisation that survives a half-precision ``param_dtype``.

    Flax's runs the QR decomposition in the param dtype, and CPU LAPACK has no
    half-precision QR kernel, so this one decomposes in single precision and casts after.
    """
    orthogonalise = nn.initializers.orthogonal(gain)

    def initialise(key: jax.Array, shape: Shape, dtype: Dtype = jnp.float32) -> jax.Array:
        return orthogonalise(key, shape, jnp.float32).astype(dtype)

    return initialise
