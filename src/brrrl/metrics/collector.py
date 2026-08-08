from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from etils import epath

from brrrl.metrics.logger import Logger
from brrrl.types import Metrics


class Origin(StrEnum):
    """Which component emitted the metrics. Routes to a chain and names the namespace."""

    ACTOR = "actor"
    LEARNER = "learner"
    EVAL = "eval"


class MetricRecorder(Protocol):
    def record(self, metrics: Metrics, transitions: int = 0) -> None:
        """Report metrics accounting for ``transitions`` environment steps."""
        ...

    def record_artifact(self, path: epath.Path, directory: str | None = None) -> None:
        """File the artifact at ``path`` under ``directory``. Carries no Step; it names a place, not a moment."""
        ...


@dataclass
class MetricCollector:
    """Owns the step clock and routes each Origin to its logger chain.

    Cumulative transitions are the step; only Actors pass a nonzero count.
    """

    loggers: dict[Origin, Logger]
    steps: int = 0

    def collect(self, origin: Origin, metrics: Metrics, transitions: int = 0) -> None:
        self.steps += transitions
        self.loggers[origin].log(metrics, self.steps)

    def collect_artifact(self, origin: Origin, path: epath.Path, directory: str | None = None) -> None:
        self.loggers[origin].log_artifact(path, directory)

    def close(self) -> None:
        for logger in self.loggers.values():
            logger.close()


@dataclass
class LocalMetricRecorder:
    collector: MetricCollector
    origin: Origin

    def record(self, metrics: Metrics, transitions: int = 0) -> None:
        self.collector.collect(self.origin, metrics, transitions)

    def record_artifact(self, path: epath.Path, directory: str | None = None) -> None:
        self.collector.collect_artifact(self.origin, path, directory)
