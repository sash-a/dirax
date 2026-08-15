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
type StepResult[Obs] = tuple[Obs, NDArray[np.floating], NDArray[np.bool_], NDArray[np.bool_], dict[str, Any]]
type Array = NDArray | jax.Array


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
    def version(self) -> NDArray[np.float32]:
        """The Version this Trajectory's acts ran under."""
        ...


class VersionedParams(NamedTuple):
    value: Params
    version: Version


class UpdateState(Protocol):
    @property
    def params(self) -> Params: ...


class UpdateRule[S: UpdateState, T: Trajectory](Protocol):
    def update(self, state: S, batch: T) -> tuple[S, Metrics]: ...
    def isvalid(self) -> bool: ...
