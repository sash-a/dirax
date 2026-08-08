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


class Transition(Protocol):
    @property
    def observation(self) -> ArrayTree: ...
    @property
    def action(self) -> NDArray: ...
    @property
    def reward(self) -> NDArray[np.float32]: ...
    @property
    def terminated(self) -> NDArray[np.bool_]: ...
    @property
    def truncated(self) -> NDArray[np.bool_]: ...


class Trajectory(Protocol):
    @property
    def transitions(self) -> Transition: ...


class VersionedParams(NamedTuple):
    value: Params
    version: Version


class LearnerState(Protocol):
    @property
    def params(self) -> Params: ...
