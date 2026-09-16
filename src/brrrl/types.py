from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, Protocol, Self

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


class PolicyOutput(Protocol):
    @property
    def action(self) -> Array: ...


class Transition[Obs, O: PolicyOutput](Protocol):
    """One slot. The flags describe the observation beside them; the rest describe the step out of it.

    Every leaf is an `Array` rather than the numpy type a stepper reported, because one Transition
    type is read on both sides of echo: numpy where a Roller builds it, jax inside the jitted update.
    """

    @property
    def observation(self) -> Obs: ...
    @property
    def terminated(self) -> Array: ...
    @property
    def truncated(self) -> Array: ...
    @property
    def reward(self) -> Array: ...
    @property
    def action(self) -> Array: ...
    @classmethod
    def record(cls, obs: Obs, term: Array, trunc: Array, reward: Array, policy_output: O) -> Self:
        """One slot from one step, built from what a stepper reported and what a Policy returned."""
        ...


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
    def init(self, key: PRNGKey) -> S: ...
