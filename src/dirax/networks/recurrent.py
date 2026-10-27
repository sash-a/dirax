from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, final

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.typing import Dtype

from dirax.types import Array

type Carry = Any

# The cell's params are created once and broadcast, so their key must not be split per step.
UNSPLIT_PARAMS: Mapping[Any, bool] = {"params": False}


@final
class ResetRecurrence(nn.Module):
    """A recurrent cell scanned over the time axis, whose carry does not cross an episode.

    Wherever a done is set, the outgoing carry is replaced by a freshly-initialised one,
    so hidden state cannot leak out of one episode and into the next. We do it this way
    because we use next step auto-reset where a `done` step gets the final observation.

    Resets arrive as an array, which is expected to be `terminated | truncated`

    Attributes:
        hidden_size: Width of the carry.
        cell: Recurrent cell type to scan. Its ``carry_init`` is what a reset lands on.
        param_dtype: Dtype params are stored in.
        dtype: Dtype the cell computes in, promoted from the inputs and params when None.
    """

    hidden_size: int
    cell: Callable[..., nn.RNNCellBase] = nn.GRUCell
    param_dtype: Dtype = jnp.float32
    dtype: Dtype | None = None

    @nn.nowrap
    def initialize_carry(self, prefix: tuple[int, ...]) -> Carry:
        """The carry a rollout starts from, and the one a reset lands on."""
        detached = self.cell(features=self.hidden_size, param_dtype=self.param_dtype, dtype=self.dtype, parent=None)
        return detached.initialize_carry(jax.random.key(0), (*prefix, self.hidden_size))

    @partial(nn.scan, variable_broadcast="params", split_rngs=UNSPLIT_PARAMS, in_axes=0, out_axes=0)
    @nn.compact
    def __call__(self, carry: Carry, inputs: jax.Array, dones: Array) -> tuple[Carry, jax.Array]:
        carry, emb = self.cell(features=self.hidden_size, param_dtype=self.param_dtype, dtype=self.dtype)(carry, inputs)
        fresh = self.initialize_carry(dones.shape)
        carry = jax.tree.map(lambda init, curr: jnp.where(dones[..., None], init, curr), fresh, carry)
        return carry, emb
