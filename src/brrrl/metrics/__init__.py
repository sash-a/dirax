from brrrl.metrics.collector import LocalMetricRecorder, MetricCollector, MetricRecorder, Origin, WindowedRecorder
from brrrl.metrics.logger import Logger, NamespacedProcessor


def default_logger(logger: Logger, every_records: int) -> tuple[MetricRecorder, MetricRecorder]:
    """The one-process chain: a window per Origin, in front of the collector that clocks them both.

    Shaped like `brrrl.distributed.components.metrics.default_remote_logger`, and differing only in
    which recorder the window wraps, so moving a run onto a cluster does not move where it folds.
    """
    collector = MetricCollector(
        {
            Origin.ACTOR: NamespacedProcessor("actor", logger),
            Origin.LEARNER: NamespacedProcessor("learner", logger),
        }
    )
    actor = WindowedRecorder(LocalMetricRecorder(collector, Origin.ACTOR), every_records)
    learner = WindowedRecorder(LocalMetricRecorder(collector, Origin.LEARNER), every_records)

    return actor, learner
