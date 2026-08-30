from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import ray
from etils import epath
from ray.actor import ActorHandle

from brrrl.distributed.placement.component import Behaviour
from brrrl.metrics.collector import MetricCollector, MetricRecorder, Origin, WindowedRecorder
from brrrl.metrics.logger import Logger, NamespacedProcessor
from brrrl.types import Metrics


class RayMetricCollector:
    """Ray placement of the one collector; built here so the backend is never pickled.

    Single-threaded by design: Ray serialises every call, so the backend never
    sees concurrent writers and needs no lock of its own.
    """

    RAY_OPTIONS: ClassVar[Behaviour] = {"max_concurrency": 1}

    def __init__(self, build_collector: Callable[[], MetricCollector]) -> None:
        self.collector = build_collector()

    def collect(self, origin: Origin, metrics: Metrics, num_transitions: int = 0) -> None:
        self.collector.collect(origin, metrics, num_transitions)

    def collect_artifact(self, origin: Origin, path: epath.Path, directory: str | None = None) -> None:
        self.collector.collect_artifact(origin, path, directory)

    def close(self) -> None:
        self.collector.close()


@dataclass
class RayMetricRecorder:
    collector: ActorHandle
    origin: Origin

    def record(self, metrics: Metrics, num_transitions: int = 0) -> None:
        self.collector.collect.remote(self.origin, metrics, num_transitions)

    def record_artifact(self, path: epath.Path, directory: str | None = None) -> None:
        """Only the path crosses the wire, so it must resolve on the collector's node."""
        self.collector.collect_artifact.remote(self.origin, path, directory)

    def close(self) -> None:
        """A view holds nothing to flush, and the collector behind it is the driver's to close."""


def default_remote_logger(
    build_backend: Callable[[], Logger], every_records: int
) -> tuple[MetricRecorder, MetricRecorder, ActorHandle[RayMetricCollector]]:
    """One collector for the whole run, and a windowed view of it for each process that records.

    The collector owning the step clock is what makes it a run's clock rather than one process's:
    Actors advance it with the transitions they took, and a Learner's metrics are stamped with
    where the run had got to, which is why it needs to pass no count of its own.

    The backend is built inside the collector rather than shipped to it, so a Logger holding a live
    tracking-server client is never pickled. Returns the handle beside the recorders because the
    collector outlives every one of them and is the driver's to close.
    """

    def build_collector() -> MetricCollector:
        backend = build_backend()
        return MetricCollector(
            {
                Origin.ACTOR: NamespacedProcessor("actor", backend),
                Origin.LEARNER: NamespacedProcessor("learner", backend),
            }
        )

    collector = ray.remote(RayMetricCollector).options(**RayMetricCollector.RAY_OPTIONS).remote(build_collector)
    actor = WindowedRecorder(RayMetricRecorder(collector, Origin.ACTOR), every_records)
    learner = WindowedRecorder(RayMetricRecorder(collector, Origin.LEARNER), every_records)
    return actor, learner, collector
