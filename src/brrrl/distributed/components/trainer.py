from dataclasses import dataclass
from typing import ClassVar

from echo import Server

from brrrl.distributed.placement.component import Behaviour
from brrrl.metrics.collector import MetricRecorder
from brrrl.params.params import ParamSink
from brrrl.types import Trajectory, UpdateRule, UpdateState


@dataclass
class Trainer[S: UpdateState, T: Trajectory]:
    """Pulls stacked trajectories off echo, updates, publishes params."""

    RAY_OPTIONS: ClassVar[Behaviour] = {"max_restarts": 0}

    rule: UpdateRule[S, T]
    server: Server
    sink: ParamSink
    metrics: MetricRecorder

    def run(self) -> None:
        raise NotImplementedError
