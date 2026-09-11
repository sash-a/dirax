from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from etils import epath

from brrrl.metrics.accumulator import MetricAccumulator
from brrrl.metrics.logger import Logger
from brrrl.types import Metrics
from brrrl.utils.duration import Duration, Schedule


class Origin(StrEnum):
    """Which component emitted the metrics. Routes to a chain and names the namespace."""

    ACTOR = "actor"
    LEARNER = "learner"
    EVAL = "eval"


class MetricRecorder(Protocol):
    def record(self, metrics: Metrics, num_transitions: int = 0) -> None:
        """Report metrics accounting for ``num_transitions`` environment steps."""
        ...

    def record_artifact(self, path: epath.Path, directory: str | None = None) -> None:
        """File the artifact at ``path`` under ``directory``. Carries no Step; it names a place, not a moment."""
        ...

    def close(self) -> None:
        """Flush whatever this recorder is holding.

        Only what this end owns: a collector shared by every recording process is closed by the
        driver that placed it, never by one of the nine processes that happens to finish first.
        """
        ...


@dataclass
class MetricCollector:
    """Owns the step clock and routes each Origin to its logger chain.

    Cumulative transitions are the step; only Actors pass a nonzero count.
    """

    loggers: dict[Origin, Logger]
    steps: int = 0

    def collect(self, origin: Origin, metrics: Metrics, num_transitions: int = 0) -> None:
        self.steps += num_transitions
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

    def record(self, metrics: Metrics, num_transitions: int = 0) -> None:
        self.collector.collect(self.origin, metrics, num_transitions)

    def record_artifact(self, path: epath.Path, directory: str | None = None) -> None:
        self.collector.collect_artifact(self.origin, path, directory)

    def close(self) -> None:
        """The collector is this process's own, so closing the recorder closes it."""
        self.collector.close()


class WindowedRecorder:
    """Folds records and forwards their summary once every ``every`` of them."""

    def __init__(self, recorder: MetricRecorder, every: Duration, verbose: bool = False, transitions: int = 0):
        self.recorder = recorder
        self.every = Schedule(every)
        self.verbose = verbose
        self.transitions = transitions
        self.accumulators = {}

    def record(self, metrics: Metrics, num_transitions: int = 0) -> None:
        for key, values in metrics.items():
            self.accumulators.setdefault(key, MetricAccumulator()).fold(values)
        self.transitions += num_transitions

        if self.every.due():
            self.flush()

    def record_artifact(self, path: epath.Path, directory: str | None = None) -> None:
        """Artifacts have nothing to fold, so they skip the window."""
        self.recorder.record_artifact(path, directory)

    def flush(self) -> None:
        """Forwards the window, carrying the transitions it spanned so the step clock stays exact."""
        accumulators, self.accumulators = self.accumulators, {}
        transitions, self.transitions = self.transitions, 0
        summary = {
            f"{key}/{stat}": value
            for key, accumulator in accumulators.items()
            for stat, value in accumulator.summary(self.verbose).items()
        }
        if summary or transitions:
            self.recorder.record(summary, transitions)

    def close(self) -> None:
        """Flushes the partial window and closes what it wraps."""
        self.flush()
        self.recorder.close()
