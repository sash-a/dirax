import logging
from collections.abc import Callable
from typing import ClassVar

import ray
from echo import TcpClient
from ray.actor import ActorHandle
from ray.exceptions import RayActorError, RayTaskError

from dirax.acting.roller import Roller
from dirax.distributed.placement.component import Behaviour
from dirax.metrics.collector import MetricRecorder
from dirax.metrics.timer import AggregatedElapsed, Timer

logger = logging.getLogger(__name__)


class Actor:
    """One thread per roller. Rollers are built here, not shipped here."""

    RAY_OPTIONS: ClassVar[Behaviour] = {"max_concurrency": 2, "max_restarts": -1}
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

        self.timer = Timer(AggregatedElapsed)
        self.stop = False

    # TODO: how to support inference servers with actors running in the background
    @ray.method
    def run(self) -> None:
        try:
            while not self.stop:
                with self.timer("rollout_time"):
                    trajectory, metrics = self.roller.rollout()
                with self.timer("replay_send_time"):
                    self.replay_client.send(trajectory)  # ty: ignore[invalid-argument-type]

                metrics = {**metrics, **self.timer.dump()}
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


class ActorMonitor:
    RAY_OPTIONS: ClassVar[Behaviour] = {"max_concurrency": 2}
    """``run`` holds a thread for the life of the run, so ``close`` needs the second one."""

    def __init__(self, actors: list[ActorHandle[Actor]], max_restarts: int, metric_recorder: MetricRecorder):
        self.actors = actors
        self.remaining_restarts = {actor: max_restarts for actor in actors}
        self.metric_recorder = metric_recorder
        self.num_restarts = 0
        self.stop = False

    @ray.method
    def run(self) -> None:
        futures = {actor.run.remote(): actor for actor in self.actors}
        errors: list[Exception] = []

        while futures:
            done, _ = ray.wait(list(futures), num_returns=1)
            actor = futures.pop(done[0])
            try:
                ray.get(done[0])
            except (RayTaskError, RayActorError) as e:
                errors.append(e)
                if self.stop:  # we are done, don't restart the actor
                    continue
                # exceeded max restarts, kill all actors and catch any of their errors
                if self.remaining_restarts[actor] == 0:
                    self.close()
                    continue

                self.remaining_restarts[actor] -= 1
                self.num_restarts += 1
                self.metric_recorder.record({"actor_restarts": self.num_restarts})
                logger.warning("Restarting actor %s. It died with %s: %s", actor, type(e).__name__, e)
                if isinstance(e, RayTaskError):
                    ray.kill(actor, no_restart=False)
                futures[actor.run.remote()] = actor

        if errors:
            raise ExceptionGroup("actors failed", errors)

    @ray.method
    def close(self) -> list[ActorHandle]:
        self.stop = True
        return [actor.close.remote() for actor in self.actors]
