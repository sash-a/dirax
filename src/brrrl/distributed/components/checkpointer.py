from collections.abc import Callable
from dataclasses import dataclass

import jax
import ray
from etils import epath
from ray.actor import ActorHandle

from brrrl.checkpointing.checkpointer import Checkpointer, Config, State
from brrrl.types import Metrics


@ray.remote
class CheckpointStore:
    """The one place checkpoints land; learner writes, a resume reads.

    Its own actor so a learner posts a state and goes back to stepping while the write lands.
    Single-threaded by design: Ray serialises every call, so saves land in the order they were posted.
    Built from a factory so the backend, which owns files and threads, is never pickled.
    """

    def __init__(self, build_checkpointer: Callable[[], Checkpointer]) -> None:
        self.checkpointer = build_checkpointer()

    def save(
        self,
        step: int,
        state: State,
        metrics: Metrics | None = None,
        *,
        metadata: Config | None = None,
        force: bool = False,
    ) -> bool:
        return self.checkpointer.save(step, state, metrics, metadata=metadata, force=force)

    def metadata(self, step: int | None = None) -> Config | None:
        return self.checkpointer.metadata(step)

    def load(self, step: int | None = None, abstract: State | None = None) -> State:
        return self.checkpointer.load(step, abstract)

    def directory(self) -> epath.Path:
        return self.checkpointer.directory

    def latest(self) -> int | None:
        return self.checkpointer.latest

    def best(self) -> int | None:
        return self.checkpointer.best

    def wait(self) -> None:
        self.checkpointer.wait()

    def close(self) -> None:
        self.checkpointer.close()


@dataclass
class RayCheckpointer:
    """A learner's view of the store: saves are posted and forgotten, reads block.

    Properties the trait reads off an attribute cost a round trip here, so a resume reads them once
    rather than per step.
    """

    store: ActorHandle

    def save(
        self,
        step: int,
        state: State,
        metrics: Metrics | None = None,
        *,
        metadata: Config | None = None,
        force: bool = False,
    ) -> None:
        """Post ``state`` to be written at ``step``, subject to the store's save policy.

        Returns nothing where the trait returns whether the policy admitted the step: waiting on that
        answer is the stall this actor exists to avoid, and ``latest`` reports it after the fact.

        Args:
            step: Step to file the checkpoint under. Steps at or below the latest are skipped.
            state: One pytree per item the store was built with, keyed by item.
            metrics: Values describing this checkpoint, stored beside it and used to rank it.
            metadata: Bookkeeping filed with this checkpoint, read back by ``metadata`` without
                restoring the arrays.
            force: Save regardless of the interval, for a run's final checkpoint.
        """
        # One transfer off device for both: arrays cannot cross the wire, and the store cannot
        # decline the step until they have.
        state, metrics = jax.device_get((state, metrics))
        self.store.save.remote(step, state, metrics, metadata=metadata, force=force)

    def metadata(self, step: int | None = None) -> Config | None:
        """Bookkeeping filed with the checkpoint at ``step``, latest when None, without reading arrays."""
        return ray.get(self.store.metadata.remote(step))

    def load(self, step: int | None = None, abstract: State | None = None) -> State:
        """Read back a checkpoint, blocking until it arrives.

        Args:
            step: Step to read, defaulting to the latest saved.
            abstract: Pytrees of ``jax.ShapeDtypeStruct`` keyed by item, naming what to restore onto.
                Also selects which items are read, so pass one to leave the rest on the store's disk.

        Returns:
            One pytree per item read, keyed by item, on the caller's host.
        """
        return ray.get(self.store.load.remote(step, abstract))

    @property
    def directory(self) -> epath.Path:
        """Resolves on the store's node, so file it as an Artifact from a collector that shares one."""
        return ray.get(self.store.directory.remote())

    @property
    def latest(self) -> int | None:
        return ray.get(self.store.latest.remote())

    @property
    def best(self) -> int | None:
        return ray.get(self.store.best.remote())

    def wait(self) -> None:
        """Block until posted saves have landed; the store takes them in order, so this drains the queue."""
        ray.get(self.store.wait.remote())

    def close(self) -> None:
        ray.get(self.store.close.remote())
