"""Doubles the Components share, the wait a test does for a thread, and what replication means.

A Component is communication glue and nothing else, so what is worth asserting about one is what
it did to its collaborators. Every collaborator that matters lives in another process on a real
run, which is what makes a double the only place their side of the exchange is visible at all.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
import pytest
from etils import epath
from numpy.typing import ArrayLike

from brrrl.types import Metrics

SETTLE = 5.0
"""How long to let a background thread reach a state before calling it stuck. Nothing races it."""


@dataclass
class Recorded:
    """A `brrrl.metrics.collector.MetricRecorder` that keeps what it was handed.

    Stands in for the run's one collector, which is a Ray actor on every path that matters. What a
    Component decided — which metrics, how many transitions it accounts them to, and whether it let
    go of the recorder on its way out — is only readable from this end.
    """

    records: list[tuple[dict[str, ArrayLike], int]] = field(default_factory=list)
    artifacts: list[tuple[epath.Path, str | None]] = field(default_factory=list)
    closes: int = 0

    def record(self, metrics: Metrics, num_transitions: int = 0) -> None:
        self.records.append((dict(metrics), num_transitions))

    def record_artifact(self, path: epath.Path, directory: str | None = None) -> None:
        self.artifacts.append((path, directory))

    def close(self) -> None:
        self.closes += 1

    @property
    def transitions(self) -> list[int]:
        """What each record was accounted to, which is the run's x-axis where it is not zero."""
        return [num_transitions for _, num_transitions in self.records]

    @property
    def keys(self) -> list[list[str]]:
        return [sorted(metrics) for metrics, _ in self.records]


@pytest.fixture
def recorder() -> Recorded:
    return Recorded()


def settled(reached: Callable[[], bool], timeout: float = SETTLE) -> None:
    """Waits for a background thread to get where it is going, so the assertion after this is about
    what it did rather than about which thread got there first."""
    deadline = time.monotonic() + timeout
    while not reached() and time.monotonic() < deadline:
        time.sleep(0.01)


def replicated(tree: Any, over: int) -> bool:
    """Every leaf of ``tree`` held whole by all ``over`` devices, with every copy the same.

    ``is_fully_replicated`` is not this claim and cannot stand in for it: it is True of a
    `SingleDeviceSharding` as well, so a parameter that never left the device it was built on
    satisfies it and a state that was never placed reads as replicated. What separates the two is
    how many devices are holding a copy and how much of the array each of them holds, which is
    what this counts.
    """
    leaves: list[jax.Array] = jax.tree.leaves(tree)
    return all(len(leaf.addressable_shards) == over for leaf in leaves) and all(
        shard.data.shape == leaf.shape and np.array_equal(shard.data, leaf.addressable_shards[0].data)
        for leaf in leaves
        for shard in leaf.addressable_shards
    )
