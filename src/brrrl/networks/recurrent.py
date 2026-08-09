from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, final

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.typing import Dtype

type Carry = Any

# The cell's params are created once and broadcast, so their key must not be split per step.
UNSPLIT_PARAMS: Mapping[Any, bool] = {"params": False}


@final
class ResetRecurrence(nn.Module):
    """A recurrent cell scanned over the time axis, whose carry does not cross an episode.

    Wherever a reset is set, the incoming carry is replaced by a freshly-initialised one
    before the cell steps, so hidden state cannot leak out of one episode and into the next
    inside a Trajectory. Resets arrive as an array, which the algorithm builds from the
    terminal flags a Transition already carries, so no Trajectory gains a field for this.

    Learn-time shape only: ``inputs`` and ``resets`` lead with ``(T, E)``. How hidden
    state crosses the act boundary is a separate open question, and nothing here decides it.

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
    def initialize_carry(self, num_envs: int) -> Carry:
        """The carry a rollout starts from, and the one a reset lands on.

        Deterministic: the cell's ``carry_init`` is zeros, so the key it takes goes unused.
        ``parent=None`` keeps this cell out of the param tree — it only reads an initialiser.
        """
        detached = self.cell(features=self.hidden_size, param_dtype=self.param_dtype, dtype=self.dtype, parent=None)
        return detached.initialize_carry(jax.random.key(0), (num_envs, self.hidden_size))

    @partial(nn.scan, variable_broadcast="params", split_rngs=UNSPLIT_PARAMS, in_axes=0, out_axes=0)
    @nn.compact
    def __call__(self, carry: Carry, inputs: jax.Array, resets: jax.Array) -> tuple[Carry, jax.Array]:
        fresh = self.initialize_carry(resets.shape[0])
        started = jax.tree.map(lambda new, held: jnp.where(resets[:, None], new, held), fresh, carry)
        return self.cell(features=self.hidden_size, param_dtype=self.param_dtype, dtype=self.dtype)(started, inputs)
