from dirax.metrics.collector import LocalMetricRecorder, MetricCollector, MetricRecorder, Origin, WindowedRecorder
from dirax.metrics.logger import Logger, NamespacedProcessor
from dirax.utils.duration import Duration


def default_logger(logger: Logger, every: Duration) -> tuple[MetricRecorder, MetricRecorder]:
    """The one-process chain: a window per Origin, in front of the collector that clocks them both.

    Shaped like `dirax.distributed.components.metrics.default_remote_logger`, and differing only in
    which recorder the window wraps, so moving a run onto a cluster does not move where it folds.

    One ``every`` serves both windows: a Duration answers a question rather than holding a
    position, and each recorder keeps the flush it last came round on.
    """
    collector = MetricCollector(
        {
            Origin.ACTOR: NamespacedProcessor("actor", logger),
            Origin.LEARNER: NamespacedProcessor("learner", logger),
        }
    )
    actor = WindowedRecorder(LocalMetricRecorder(collector, Origin.ACTOR), every)
    learner = WindowedRecorder(LocalMetricRecorder(collector, Origin.LEARNER), every)

    return actor, learner
