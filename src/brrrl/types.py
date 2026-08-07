from collections.abc import Iterable, Mapping
from typing import NamedTuple, Protocol

import jax
import numpy as np
from flax.typing import VariableDict
from numpy.typing import NDArray

type Params = VariableDict
type PRNGKey = jax.Array
type Version = int
type Metrics = dict[str, Iterable[float]]
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
