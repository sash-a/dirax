from collections.abc import Callable
from typing import ClassVar

import jax
import numpy as np

from brrrl.distributed.components.echo import ReplayServer
from brrrl.distributed.components.update_rule import DistributedUpdateRule
from brrrl.distributed.placement.component import Behaviour
from brrrl.metrics.collector import MetricRecorder
from brrrl.metrics.timer import AggregatedElapsed, Timer
from brrrl.params.params import ParamSink
from brrrl.types import Trajectory, UpdateState


class Trainer[S: UpdateState, T: Trajectory]:
    """Pulls stacked trajectories from an iterator, updates, publishes params."""

    RAY_OPTIONS: ClassVar[Behaviour] = {"max_restarts": 0}

    def __init__(
        self,
        num_updates: int,
        state: S,
        rule: DistributedUpdateRule[S, T],
        replay: ReplayServer,
        # TODO: give sink a send period or make a windowed sink
        # does this need to be a factory, it's gonna hold a ref to store
        make_param_sink: Callable[[], ParamSink],
        metric_recorder: MetricRecorder,
    ):
        self.num_updates = num_updates
        self.rule = rule

        self.param_sink = make_param_sink()
        self.param_sink.publish(state.params)

        self.metric_recorder = metric_recorder
        self.timer = Timer(AggregatedElapsed)

        model_sharding, data_sharding = rule.shardings()
        self.update_fn = rule.compiled(model_sharding, data_sharding)

        self.replay_servers = replay.servers(len(data_sharding.device_set))
        self.dataset = replay.dataset(self.replay_servers, data_sharding)

        # TODO: will this automatically replicate?
        self.state = jax.device_put(state, model_sharding)

    def run(self) -> None:
        for i in range(self.num_updates):
            with self.timer("traj_wait_time"):
                trajectory, roller_version, replay_metrics = next(self.dataset)

            with self.timer("update_dispatch_time"):
                self.state, learn_metrics = self.rule.update(self.state, trajectory)

            # TODO: ideally we overlap this publish with the update.
            # To do this we'd need to publish the previous params
            # Or we could put this work on a thread
            with self.timer("param_publish_time"):
                trainer_version = self.param_sink.publish(self.state.params)

            # TODO: checkpointing
            with self.timer("checkpointing_time"):
                ...

            versions = {"staleness": np.mean(trainer_version - roller_version)}
            # TODO: check if windowed metric recording will handle lists and scalars
            self.metric_recorder.record({**replay_metrics, **learn_metrics, **versions, **self.timer.dump()})
