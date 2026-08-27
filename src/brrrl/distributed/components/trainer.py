from typing import Callable, ClassVar, Iterator

import echo

from brrrl.distributed.placement.component import Behaviour
from brrrl.metrics.collector import MetricRecorder
from brrrl.metrics.timer import AggregatedElapsed, Timer
from brrrl.params.params import ParamSink
from brrrl.types import Trajectory, UpdateRule, UpdateState


class Trainer[S: UpdateState, T: Trajectory]:
    """Pulls stacked trajectories from an iterator, updates, publishes params."""

    RAY_OPTIONS: ClassVar[Behaviour] = {"max_restarts": 0}

    def __init__(
        self,
        num_updates: int,
        state: S,
        rule: UpdateRule[S, T],
        # TODO: dataset + sink probably need to be factories
        # TODO: strong type the metrics coming from dataset
        # TODO: make a Trajectory store/sink/source
        make_echo_server: Callable[[], echo.Server],
        make_dataset: Callable[[echo.Server], Iterator[echo.Sample]],
        # TODO: give sink a send period or make a windowed sink
        # does this need to be a factory, it's gonna hold a ref to store
        make_param_sink: Callable[[], ParamSink],
        metric_recorder: MetricRecorder,
    ):
        self.num_updates = num_updates
        self.state = state
        self.rule = rule
        self.echo_server = make_echo_server()
        self.dataset = make_dataset(self.echo_server)
        self.param_sink = make_param_sink()
        self.metric_recorder = metric_recorder
        self.timer = Timer(AggregatedElapsed)

    def run(self) -> None:
        for i in range(self.num_updates):
            with self.timer("traj_wait_time"):
                batch, replay_metrics = next(self.dataset)

            with self.timer("update_dispatch_time"):
                self.state, learn_metrics = self.rule.update(self.state, batch)

            # TODO: ideally we overlap this publish with the update.
            # To do this we'd need to publish the previous params
            # Or we could put this work on a thread
            with self.timer("param_publish_time"):
                self.param_sink.publish(self.state.params)

            # TODO: checkpointing
            with self.timer("checkpoint_time"):
                ...

            # TODO: check if windowed metric recording will handle lists and scalars
            self.metric_recorder.record(replay_metrics | learn_metrics | self.timer.dump())
