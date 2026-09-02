"""Doubles the Components share, and the wait a test does for a thread one of them holds.

A Component is communication glue and nothing else, so what is worth asserting about one is what
it did to its collaborators. Every collaborator that matters lives in another process on a real
run, which is what makes a double the only place their side of the exchange is visible at all.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

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
