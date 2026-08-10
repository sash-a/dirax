from dataclasses import dataclass
from typing import ClassVar

from echo import Server

from brrrl.acting.params import ParamSink
from brrrl.distributed.placement.component import Behaviour
from brrrl.learning.learner import Learner
from brrrl.metrics.collector import MetricRecorder
from brrrl.types import LearnerState, Trajectory


@dataclass
class Trainer[S: LearnerState, T: Trajectory]:
    """Pulls stacked trajectories off echo, updates, publishes params."""

    RAY_OPTIONS: ClassVar[Behaviour] = {"max_restarts": 0}

    learner: Learner[S, T]
    server: Server
    sink: ParamSink
    metrics: MetricRecorder

    def run(self) -> None:
        raise NotImplementedError
