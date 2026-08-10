from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from etils import epath
from ray.actor import ActorHandle

from brrrl.distributed.placement.component import Behaviour
from brrrl.metrics.collector import MetricCollector, Origin
from brrrl.types import Metrics


class RayMetricCollector:
    """Ray placement of the one collector; built here so the backend is never pickled.

    Single-threaded by design: Ray serialises every call, so the backend never
    sees concurrent writers and needs no lock of its own.
    """

    RAY_OPTIONS: ClassVar[Behaviour] = {"max_concurrency": 1}

    def __init__(self, build_collector: Callable[[], MetricCollector]) -> None:
        self.collector = build_collector()

    def collect(self, origin: Origin, metrics: Metrics, transitions: int = 0) -> None:
        self.collector.collect(origin, metrics, transitions)

    def collect_artifact(self, origin: Origin, path: epath.Path, directory: str | None = None) -> None:
        self.collector.collect_artifact(origin, path, directory)

    def close(self) -> None:
        self.collector.close()


@dataclass
class RayMetricRecorder:
    collector: ActorHandle
    origin: Origin

    def record(self, metrics: Metrics, transitions: int = 0) -> None:
        self.collector.collect.remote(self.origin, metrics, transitions)

    def record_artifact(self, path: epath.Path, directory: str | None = None) -> None:
        """Only the path crosses the wire, so it must resolve on the collector's node."""
        self.collector.collect_artifact.remote(self.origin, path, directory)
