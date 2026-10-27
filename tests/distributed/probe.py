"""A Component the preflight tests place, in its own module because a worker imports it.

A worker's environment carries dirax's runtime dependencies and nothing else — `uv run` gives
it the project, not the `dev` extra — so anything it imports must hold for a deployed Actor too.
"""

import os
from typing import ClassVar

import ray

from dirax.distributed.placement.component import Behaviour

WATCHED = ("JAX_PLATFORMS", "DIRAX_JOB_WIDE")


class Probe:
    """A Component that reports what it was actually given."""

    RAY_OPTIONS: ClassVar[Behaviour] = {}

    def report(self) -> tuple[dict[str, float], dict[str, str]]:
        assigned = ray.get_runtime_context().get_assigned_resources()
        return assigned, {name: os.environ[name] for name in WATCHED if name in os.environ}
