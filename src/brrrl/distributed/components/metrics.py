from collections.abc import Callable
from dataclasses import dataclass

import ray
from ray.actor import ActorHandle

from brrrl.metrics.logger import Logger, Source
from brrrl.types import Metrics


@ray.remote
class MetricsStore:
    """The one place metrics are written from; the step clock lives here.

    Single-threaded by design: Ray serialises every call, so the backend never
    sees concurrent writers and needs no lock of its own.
    """

    def __init__(self, build_loggers: Callable[[], dict[Source, Logger]]) -> None:
        self.loggers = build_loggers()
        self.steps = 0

    def log(self, source: Source, metrics: Metrics, transitions: int = 0) -> None:
        self.steps += transitions
        self.loggers[source].log(metrics, self.steps)

    def close(self) -> None:
        for logger in self.loggers.values():
            logger.close()


@dataclass
class RayMetricsSink:
    store: ActorHandle
    source: Source

    def log(self, metrics: Metrics, transitions: int = 0) -> None:
        self.store.log.remote(self.source, metrics, transitions)
