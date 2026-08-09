from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax
import pytest

NUM_STEPS = 3
NUM_LANES = 4


@pytest.fixture(params=[(NUM_LANES,), (NUM_STEPS, NUM_LANES)], ids=["act", "learn"])
def leading(request: pytest.FixtureRequest) -> tuple[int, ...]:
    """The two leading-axis layouts every primitive must accept unchanged.

    ``(Lane,)`` is what a Roller applies at act time; ``(time, Lane)`` is what a Learner
    applies over a Trajectory. A primitive that needs reshaping between them fails here.
    """
    return request.param


@pytest.fixture
def apply_module() -> Callable[..., Any]:
    """Initialise a module on the arguments it is about to be applied to, then apply it."""

    def run(module: nn.Module, *args: Any) -> Any:
        return module.apply(module.init(jax.random.key(0), *args), *args)

    return run
