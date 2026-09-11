from collections.abc import Callable
from typing import ClassVar

import jax
import ray
from etils import epath
from ray.actor import ActorProxy

from brrrl.checkpointing.checkpointer import Checkpointer, Config
from brrrl.distributed.placement.component import Behaviour
from brrrl.types import Metrics, UpdateState
from brrrl.utils.duration import Duration, Schedule


class CheckpointStore[S: UpdateState]:
    """The one place checkpoints land; learner writes, a resume reads.

    Its own actor so a learner posts a state and goes back to stepping while the write lands.
    Single-threaded by design: Ray serialises every call, so saves land in the order they were posted.
    Built from a factory so the backend, which owns files and threads, is never pickled.
    """

    RAY_OPTIONS: ClassVar[Behaviour] = {
        "max_concurrency": 1,
        "max_task_retries": 0,
        "allow_out_of_order_execution": False,
    }

    def __init__(self, build_checkpointer: Callable[[], Checkpointer[S]]) -> None:
        self.checkpointer = build_checkpointer()

    @ray.method
    def save(self, step: int, state: S, metrics: Metrics | None, metadata: Config | None, force: bool) -> bool:
        return self.checkpointer.save(step, state, metrics, metadata=metadata, force=force)

    @ray.method
    def metadata(self, step: int | None) -> Config | None:
        return self.checkpointer.metadata(step)

    @ray.method
    def load(self, step: int | None, abstract: S | None) -> S:
        return self.checkpointer.load(step, abstract)

    @ray.method
    def directory(self) -> epath.Path:
        return self.checkpointer.directory

    @ray.method
    def latest(self) -> int | None:
        return self.checkpointer.latest

    @ray.method
    def best(self) -> int | None:
        return self.checkpointer.best

    @ray.method
    def wait(self) -> None:
        self.checkpointer.wait()

    @ray.method
    def close(self) -> None:
        self.checkpointer.close()


class RayCheckpointer[S: UpdateState]:
    """A learner's view of the store: saves are posted and forgotten, reads block.

    Args:
        store: the ray actor backend store
        every: how much has to pass between saves. The wrapped checkpointer's own save policy
          still applies behind this one, so leave it saving every step unless both are wanted.
    """

    def __init__(self, store: ActorProxy[CheckpointStore[S]], every: Duration):
        self.store = store
        self.every = Schedule(every)
        self.save_futures = []

    def save(
        self,
        step: int,
        state: S,
        metrics: Metrics | None = None,
        *,
        metadata: Config | None = None,
        force: bool = False,
    ) -> None:
        """Post ``state`` to be written at ``step``, subject to the store's save policy.

        Args:
            step: Step to file the checkpoint under. Steps at or below the latest are skipped.
            state: The learner's state, as one pytree.
            metrics: Values describing this checkpoint, stored beside it and used to rank it.
            metadata: Bookkeeping filed with this checkpoint, read back by ``metadata`` without
                restoring the arrays.
            force: Save regardless of the interval, for a run's final checkpoint.
        """
        # Only do the D2H and remote call on required steps
        if self.every.due() or force:
            # One transfer off device for both: arrays cannot cross the wire, and the store cannot
            # decline the step until they have.
            state, metrics = jax.device_get((state, metrics))
            self._check_for_save_errors()
            save_future = self.store.save.remote(step, state, metrics, metadata, force)
            self.save_futures.append(save_future)

    def metadata(self, step: int | None = None) -> Config | None:
        """Bookkeeping filed with the checkpoint at ``step``, latest when None, without reading arrays."""
        return ray.get(self.store.metadata.remote(step))

    def load(self, step: int | None = None, abstract: S | None = None) -> S:
        """Read back a checkpoint, blocking until it arrives.

        Args:
            step: Step to read, defaulting to the latest saved.
            abstract: The state's shape as ``jax.ShapeDtypeStruct`` leaves, naming what to restore
                onto. Leaves marked ``ocp.PLACEHOLDER`` are left on the store's disk.

        Returns:
            The state, on the caller's host.
        """
        return ray.get(self.store.load.remote(step, abstract))  # ty: ignore[invalid-argument-type]

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
        self._check_for_save_errors(timeout=None)
        ray.get(self.store.wait.remote())

    def close(self) -> None:
        self._check_for_save_errors(timeout=None)
        ray.get(self.store.close.remote())

    def _check_for_save_errors(self, timeout: int | None = 0) -> None:
        # Bubble up the errors from the previous saves if they exist
        if not self.save_futures:
            return

        ready, self.save_futures = ray.wait(self.save_futures, num_returns=len(self.save_futures), timeout=timeout)
        ray.get(ready)
