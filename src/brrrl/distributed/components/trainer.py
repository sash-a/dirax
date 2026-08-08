from dataclasses import dataclass

from echo import Server

from brrrl.acting.params import ParamSink
from brrrl.learning.learner import Learner
from brrrl.metrics.collector import MetricRecorder
from brrrl.types import LearnerState, Trajectory


@dataclass
class Trainer[S: LearnerState, T: Trajectory]:
    """Pulls stacked trajectories off echo, updates, publishes params."""

    learner: Learner[S, T]
    server: Server
    sink: ParamSink
    metrics: MetricRecorder

    def run(self) -> None:
        raise NotImplementedError
