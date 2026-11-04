import logging
import time
import uuid
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

PING_INTERVAL = 0.1


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
        self.instance = uuid.uuid4().hex

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
    def ping(self) -> str:
        """Names this incarnation, so a caller can tell a restarted actor from the one it replaced."""
        return self.instance

    @ray.method
    def close(self) -> None:
        """Asks the loop to stop after the rollout it is on, and returns without waiting for it."""
        self.stop = True


class ActorMonitor:
    RAY_OPTIONS: ClassVar[Behaviour] = {"max_concurrency": 2}
    """``run`` holds a thread for the life of the run, so ``close`` needs the second one."""

    def __init__(
        self,
        actors: list[ActorHandle[Actor]],
        max_restarts: int,
        metric_recorder: MetricRecorder,
        restart_timeout: float = 300.0,
    ):
        """
        Monitors the actors `run` methods for crashes. Restarts actors and surfaces errors if crashes occur.
        Args:
            actors: a list of `Actor`s to monitor.
            max_restarts: the max number of restarts each actor has before killing the experiment.
            metric_recorder: used to report the number of restarts.
            restart_timeout: if an actor has not restarted within this time then kill the experiment.
        """
        self.actors = actors
        self.remaining_restarts = {actor: max_restarts for actor in actors}
        self.metric_recorder = metric_recorder
        self.restart_timeout = restart_timeout
        self.num_restarts = 0
        self.stop = False

    @ray.method
    def run(self) -> None:
        """Starts all actors, monitors them for crashes and restarts them up to `max_restarts`."""
        try:
            futures = {actor.run.remote(): actor for actor in self.actors}
            errors: list[Exception] = []

            while futures:
                done, _ = ray.wait(list(futures), num_returns=1)
                actor = futures.pop(done[0])
                try:
                    ray.get(done[0])
                except (RayTaskError, RayActorError) as e:
                    if self.stop or self.remaining_restarts[actor] == 0:
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
                    previous = None
                    if isinstance(e, RayTaskError):
                        previous = self._id_of(actor)
                        ray.kill(actor, no_restart=False)
                    if not self._await_restart(actor, previous):
                        errors.append(e)
                        self.close()
                        continue
                    futures[actor.run.remote()] = actor
        finally:
            for actor in self.actors:
                ray.kill(actor, no_restart=True)

        if errors:
            raise ExceptionGroup("actors failed", errors)

    def _id_of(self, actor: ActorHandle[Actor]) -> str | None:
        """Get the current ID of an actor returned from its `ping` method."""
        try:
            return ray.get(actor.ping.remote())
        except RayActorError:
            return None

    def _await_restart(self, actor: ActorHandle[Actor], previous: str | None) -> bool:
        """ray.kill is asynchronous, so wait for a different ping response to confirm the restart."""
        deadline = time.monotonic() + self.restart_timeout
        while time.monotonic() < deadline:
            current = self._id_of(actor)
            if current is not None and current != previous:
                return True
            time.sleep(PING_INTERVAL)
        return False

    @ray.method
    def close(self) -> list[ActorHandle]:
        self.stop = True
        return [actor.close.remote() for actor in self.actors]
