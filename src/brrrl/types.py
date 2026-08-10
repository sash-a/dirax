from collections.abc import Mapping
from typing import Any, NamedTuple, Protocol

import jax
import numpy as np
from flax.typing import VariableDict
from numpy.typing import ArrayLike, NDArray

type Params = VariableDict
type PRNGKey = jax.Array
type Version = int
type Metrics = Mapping[str, ArrayLike]
type NestedMetrics = Mapping[str, Any]
type ArrayTree = NDArray | Mapping[str, ArrayTree] | tuple[ArrayTree, ...]
type Array = NDArray | jax.Array
"""A leaf with two residencies: a policy output is built on the device and read back on the host.

That a Roller only ever sees the host one is enforced by `Policy.act`, not by this alias.
"""


TRANSITIONS = "transitions"
VERSION = "version"
"""A Trajectory's timescale names. They are its field names too, so the accumulator rebuilds it."""


class Transition(Protocol):
    """One slot. The flags describe the observation beside them; the rest describe the step out of it."""

    @property
    def observation(self) -> ArrayTree: ...
    @property
    def terminated(self) -> NDArray[np.bool_]: ...
    @property
    def truncated(self) -> NDArray[np.bool_]: ...
    @property
    def reward(self) -> NDArray[np.floating]: ...
    @property
    def action(self) -> Array: ...


class Trajectory(Protocol):
    @property
    def transitions(self) -> Transition: ...
    @property
    def version(self) -> NDArray[np.int32]:
        """The single-item Timescale holding the one Version the whole Trajectory was acted under."""
        ...


class VersionedParams(NamedTuple):
    value: Params
    version: Version


class LearnerState(Protocol):
    @property
    def params(self) -> Params: ...
