from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, Protocol

import jax
import numpy as np
from flax.typing import VariableDict
from numpy.typing import ArrayLike, NDArray

type Params = VariableDict
type PRNGKey = jax.Array
type Metrics = Mapping[str, ArrayLike]
type NestedMetrics = Mapping[str, Any]
type ArrayTree = NDArray | Mapping[str, ArrayTree] | tuple[ArrayTree, ...]
type StepResult[Obs] = tuple[Obs, NDArray[np.floating], NDArray[np.bool_], NDArray[np.bool_], dict[str, Any]]
type Array = NDArray | jax.Array
type UpdateFn[S: UpdateState, T: Trajectory] = Callable[[S, T], tuple[S, Metrics]]


TRANSITIONS = "transitions"
"""A Trajectory's timescale name. It is its field name too, so the accumulator rebuilds it."""


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


# TODO: do we need a Trajectory anymore? Might be useful if we create things at different timescales
class Trajectory(Protocol):
    @property
    def transitions(self) -> Transition: ...


class Rollout[T: Trajectory](NamedTuple):
    """One actor's submission. The Roller stamps the version, so no user type carries it."""

    trajectory: T
    version: NDArray[np.float32]


class TrainerBatch[T: Trajectory](NamedTuple):
    """The Rollouts one update consumes: payload sharded across devices, provenance on the host."""

    trajectory: T
    version: NDArray[np.float32]
    metrics: Metrics


class VersionedParams(NamedTuple):
    value: Params
    version: int


class UpdateState(Protocol):
    @property
    def params(self) -> Params: ...


class UpdateRule[S: UpdateState, T: Trajectory](Protocol):
    def update(self, state: S, trajectory: T) -> tuple[S, Metrics]: ...
    def isvalid(self) -> bool: ...
