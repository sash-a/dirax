from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax
import pytest

NUM_STEPS = 3
NUM_ENVS = 4


@pytest.fixture(params=[(NUM_ENVS,), (NUM_STEPS, NUM_ENVS)], ids=["act", "learn"])
def leading(request: pytest.FixtureRequest) -> tuple[int, ...]:
    """The two leading-axis layouts every primitive must accept unchanged.

    ``(E,)`` is what a Roller applies at act time; ``(T, E)`` is what a Learner
    applies over a Trajectory. A primitive that needs reshaping between them fails here.
    """
    return request.param


@pytest.fixture
def apply_module() -> Callable[..., Any]:
    """Initialise a module on the arguments it is about to be applied to, then apply it."""

    def run(module: nn.Module, *args: Any) -> Any:
        return module.apply(module.init(jax.random.key(0), *args), *args)

    return run
