from brrrl.metrics.collector import LocalMetricRecorder, MetricCollector, Origin
from brrrl.metrics.logger import Logger, MultiLogger, NamespacedProcessor, WindowedProcessor


def default_local_logger(*loggers: Logger, log_every: int) -> tuple[LocalMetricRecorder, LocalMetricRecorder]:

    backend = MultiLogger(list(loggers))
    logger = WindowedProcessor(backend, log_every)
    metric_collector = MetricCollector(
        {
            Origin.ACTOR: NamespacedProcessor("actor", logger),
            Origin.LEARNER: NamespacedProcessor("learner", logger),
        }
    )
    actor_recorder = LocalMetricRecorder(metric_collector, Origin.ACTOR)
    learner_recorder = LocalMetricRecorder(metric_collector, Origin.LEARNER)

    return actor_recorder, learner_recorder


# TODO: move this to distributed and do it there
def default_remote_logger(*loggers: Logger, log_every: int): ...
