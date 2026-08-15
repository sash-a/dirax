import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import pytest

NUM_STEPS = 3
NUM_ENVS = 4

COMPILATION_CACHE = Path(__file__).parent.parent / ".jax-cache"


def pytest_configure() -> None:
    """Keep compiled kernels across runs, since compiling them is most of what this suite does.

    A driver started by ``uv run`` also has Ray rewrite every worker's interpreter to that same
    ``uv run`` command against an uploaded copy of the cwd, where uv resolves a different .venv
    and the worker cannot import ray at all. Ray's own flag turns that off, ahead of every import
    of ray, and an actor claiming a CPU the cluster plainly has is placeable again.

    One xdist worker per core would each take a CUDA context and preallocate most of the card off
    each other, so the suite is pinned to CPU; nothing here tests a kernel that needs a GPU.
    ``JAX_PLATFORMS`` is read as jax is imported and so cannot be set from here, but a platform
    chosen explicitly still wins.

    A primitive applied eagerly compiles the same kernels every run, and JAX caches those in
    process only. Both thresholds have to move for the on-disk cache to hold anything: it ships
    tuned for training steps and skips whatever compiles in under a second, which is all of this.
    ``JAX_COMPILATION_CACHE_DIR`` still wins, so the cache can be moved somewhere shared.
    """
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    if not jax.config.jax_platforms:
        jax.config.update("jax_platforms", "cpu")
    if not jax.config.jax_compilation_cache_dir:
        jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


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
