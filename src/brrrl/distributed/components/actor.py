from collections.abc import Callable
from typing import ClassVar

import ray
from echo import TcpClient

from brrrl.acting.roller import Roller
from brrrl.distributed.placement.component import Behaviour
from brrrl.metrics.collector import MetricRecorder


# TODO: better, non-conflicting name
class Actor:
    """One thread per roller. Rollers are built here, not shipped here."""

    RAY_OPTIONS: ClassVar[Behaviour] = {"max_concurrency": 2}
    """``run`` holds a thread for the life of the run, so ``close`` needs the second one."""

    def __init__(
        self,
        make_roller: Callable[[], Roller],
        make_replay_client: Callable[[], TcpClient],
        metric_recorder: MetricRecorder,
    ):
        self.roller = make_roller()
        self.replay_client = make_replay_client()
        self.metric_recorder = metric_recorder

        self.stop = False

    # TODO: how to support inference servers with actors running in the background
    @ray.method
    def run(self) -> None:
        try:
            while not self.stop:
                trajectory, metrics = self.roller.rollout()
                self.replay_client.send(trajectory)  # ty: ignore[invalid-argument-type]
                self.metric_recorder.record(metrics, self.roller.num_transitions)
        finally:
            self.roller.close()
            self.replay_client.close()
            self.metric_recorder.close()

    @ray.method
    def close(self) -> None:
        """Asks the loop to stop after the rollout it is on, and returns without waiting for it.

        Releasing is ``run``'s own job on its way out, because a caller here is on another thread
        and would be freeing a client from under a send still in flight.
        """
        self.stop = True
