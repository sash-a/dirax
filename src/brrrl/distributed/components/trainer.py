import time
from typing import ClassVar

import numpy as np
import ray

from brrrl.distributed.components.checkpointer import RayCheckpointer
from brrrl.distributed.components.echo import ReplayServer
from brrrl.distributed.components.update_rule import DistributedUpdateRule
from brrrl.distributed.placement.component import Behaviour
from brrrl.metrics.collector import MetricRecorder
from brrrl.metrics.timer import AggregatedElapsed, Timer
from brrrl.params.params import ParamSink
from brrrl.types import PRNGKey, Trajectory, UpdateState


class Trainer[S: UpdateState, T: Trajectory]:
    """Pulls stacked trajectories from an iterator, updates, publishes params."""

    RAY_OPTIONS: ClassVar[Behaviour] = {"max_restarts": 0}

    def __init__(
        self,
        num_updates: int,
        timesteps_per_device: int,
        key: PRNGKey,
        rule: DistributedUpdateRule[S, T],
        replay: ReplayServer,
        param_sink: ParamSink,
        metric_recorder: MetricRecorder,
        checkpointer: RayCheckpointer[S],
    ):
        mesh, model_sharding, data_sharding = rule.shardings()
        self.num_updates = num_updates
        self.timesteps_per_update = timesteps_per_device * data_sharding.mesh.devices.size
        self.state = rule.init(key, model_sharding)

        self.param_sink = param_sink
        self.param_sink.publish(self.state.params)  # Ahead of the compile: every Actor waits on this one.

        self.metric_recorder = metric_recorder
        self.timer = Timer(AggregatedElapsed)
        self.checkpointer = checkpointer

        self.update_fn = rule.compiled(mesh, model_sharding, data_sharding)

        self.replay_servers = replay.servers(list(data_sharding.mesh.devices.flat))
        self.dataset = replay.dataset(self.replay_servers, data_sharding)

    @ray.method
    def run(self) -> None:
        for i in range(self.num_updates):
            start_time = time.monotonic()
            with self.timer("traj_wait_time"):
                trajectory, roller_version, replay_metrics = next(self.dataset)

            with self.timer("update_dispatch_time"):
                self.state, learn_metrics = self.update_fn(self.state, trajectory)

            timestep = (i + 1) * self.timesteps_per_update

            # TODO: may need to put this on a thread to fully overlap with the update
            with self.timer("param_publish_time"):
                trainer_version = self.param_sink.publish(self.state.params)

            # TODO:
            # checkpointing with metrics what is useful to add here and adding return in evaluator
            with self.timer("checkpointing_time"):
                self.checkpointer.save(i, self.state, None, metadata={"timestep": timestep, "update": i + 1})

            sps = self.timesteps_per_update / (time.monotonic() - start_time)
            versions = {"staleness": np.mean(trainer_version - roller_version)}
            self.metric_recorder.record(
                {**replay_metrics, **learn_metrics, **versions, **self.timer.dump(), "learned_steps_per_second": sps}
            )

        final_timestep = self.num_updates * self.timesteps_per_update
        self.checkpointer.save(
            self.num_updates + 1,
            self.state,
            None,
            metadata={"timestep": final_timestep, "update": self.num_updates + 1},
            force=True,
        )
        self.checkpointer.wait()

    @ray.method
    def close(self) -> None:
        """Stops the replay servers and flushes what the metric window still holds.

        Called once every Actor has stopped, never before: the servers' drainer threads are what
        acknowledge an Actor's in-flight sends, and an Actor blocked in ``send`` is waiting on an
        acknowledgement rather than on anything it can be told to abandon.
        """
        for server in self.replay_servers:
            server.close()
        self.metric_recorder.close()

    @ray.method
    def get_replay_addresses(self) -> list[str]:
        """Returns one address per echo server."""
        ip = ray.util.get_node_ip_address()
        return [f"{ip}:{server.port}" for server in self.replay_servers]
