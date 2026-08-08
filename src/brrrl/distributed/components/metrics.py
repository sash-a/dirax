from collections.abc import Callable
from dataclasses import dataclass

import ray
from ray.actor import ActorHandle

from brrrl.metrics.collector import MetricCollector, Origin
from brrrl.types import Metrics


@ray.remote
class RayMetricCollector:
    """Ray placement of the one collector; built here so the backend is never pickled.

    Single-threaded by design: Ray serialises every call, so the backend never
    sees concurrent writers and needs no lock of its own.
    """

    def __init__(self, build_collector: Callable[[], MetricCollector]) -> None:
        self.collector = build_collector()

    def collect(self, origin: Origin, metrics: Metrics, transitions: int = 0) -> None:
        self.collector.collect(origin, metrics, transitions)

    def close(self) -> None:
        self.collector.close()


@dataclass
class RayMetricRecorder:
    collector: ActorHandle
    origin: Origin

    def record(self, metrics: Metrics, transitions: int = 0) -> None:
        self.collector.collect.remote(self.origin, metrics, transitions)
