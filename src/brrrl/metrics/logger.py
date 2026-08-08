from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from brrrl.metrics.accumulator import MetricAccumulator
from brrrl.types import Metrics


class Source(StrEnum):
    """Where metrics came from. Routes to a logger chain and names the namespace."""

    ACTOR = "actor"
    LEARNER = "learner"
    EVAL = "eval"


class Logger(Protocol):
    def log(self, metrics: Metrics, step: int) -> None: ...

    def close(self) -> None:
        """Flush and release. Must be idempotent: chains share a backend."""
        ...


class MetricsSink(Protocol):
    def log(self, metrics: Metrics, transitions: int = 0) -> None:
        """Record metrics accounting for ``transitions`` environment steps."""
        ...


@dataclass
class MetricCountingSink:
    """Cumulative transitions are the step. Only Actors pass a nonzero count."""

    logger: Logger
    steps: int = 0

    def log(self, metrics: Metrics, transitions: int = 0) -> None:
        self.steps += transitions
        self.logger.log(metrics, self.steps)


@dataclass
class NamespacedLogger:
    prefix: str
    logger: Logger

    def log(self, metrics: Metrics, step: int) -> None:
        self.logger.log({f"{self.prefix}/{key}": value for key, value in metrics.items()}, step)

    def close(self) -> None:
        self.logger.close()


@dataclass
class WindowedLogger:
    """Folds records and emits their summary once every ``every`` steps."""

    logger: Logger
    every: int
    verbose: bool = False
    accumulators: dict[str, MetricAccumulator] = field(default_factory=dict)
    next_flush: int = 0
    last_step: int = 0

    def log(self, metrics: Metrics, step: int) -> None:
        for key, values in metrics.items():
            self.accumulators.setdefault(key, MetricAccumulator()).fold(values)
        self.last_step = step
        if step >= self.next_flush:
            self.flush(step)

    def flush(self, step: int) -> None:
        accumulators, self.accumulators = self.accumulators, {}
        self.next_flush = (step // self.every + 1) * self.every
        summary = {
            f"{key}/{stat}": value
            for key, accumulator in accumulators.items()
            for stat, value in accumulator.summary(self.verbose).items()
        }
        if summary:
            self.logger.log(summary, step)

    def close(self) -> None:
        self.flush(self.last_step)
        self.logger.close()
