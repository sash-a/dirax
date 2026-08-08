from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, final

import orbax.checkpoint as ocp
from etils import epath

from brrrl.metrics.scalar import scalars
from brrrl.types import Metrics

type Config = Mapping[str, Any]
type State = Mapping[str, Any]


class Checkpointer(Protocol):
    def save(
        self,
        step: int,
        state: State,
        metrics: Metrics | None = None,
        *,
        metadata: Config | None = None,
        force: bool = False,
    ) -> bool:
        """Write ``state`` at ``step``.

        Args:
            step: Step to file the checkpoint under. Steps at or below the latest are skipped.
            state: One pytree per item name, keyed by item.
            metrics: Values describing this checkpoint, stored beside it and used to rank it.
            metadata: Bookkeeping filed with this checkpoint, read back without restoring it.
            force: Save regardless of the interval, for a run's final checkpoint.

        Returns:
            Whether the save policy admitted this step.
        """
        ...

    def metadata(self, step: int | None = None) -> Config | None:
        """Bookkeeping filed with the checkpoint at ``step``, latest when None.

        Args:
            step: Step to read, defaulting to the latest saved.

        Returns:
            What ``save`` filed as ``metadata``, empty when it filed none, and None when there is
            no checkpoint to read.
        """
        ...

    def load(self, step: int | None = None, abstract: State | None = None) -> State:
        """Read back a checkpoint.

        Args:
            step: Step to read, defaulting to the latest saved.
            abstract: Pytrees of ``jax.ShapeDtypeStruct`` keyed by item, naming what to restore
                onto. Also selects which items are read, so pass one to leave the rest on disk.

        Returns:
            One pytree per item read, keyed by item.
        """
        ...

    @property
    def directory(self) -> epath.Path:
        """Where checkpoints land, for filing them as an Artifact."""
        ...

    @property
    def latest(self) -> int | None: ...

    @property
    def best(self) -> int | None: ...

    def wait(self) -> None:
        """Block until in-flight saves have landed, without releasing anything."""
        ...

    def close(self) -> None:
        """Flush and release. Must be idempotent."""
        ...


def _handler_registry(items: Sequence[str]) -> ocp.DefaultCheckpointHandlerRegistry:
    handler = ocp.PyTreeCheckpointHandler()
    registry = ocp.DefaultCheckpointHandlerRegistry()
    for item in items:
        registry.add(item, ocp.args.PyTreeSave, handler)
        registry.add(item, ocp.args.PyTreeRestore, handler)
    return registry


@final
class OrbaxCheckpointer:
    """Saves a run's state as orbax checkpoints under one local directory.

    Items are fixed at construction so callers pass plain pytrees and never orbax's args vocabulary.
    """

    def __init__(
        self,
        directory: epath.PathLike,
        items: Sequence[str],
        *,
        config: Config | None = None,
        save_every: int = 1,
        keep_latest: int = 1,
        keep_best: int = 0,
        best_metric: str | None = None,
        best_mode: str = "max",
    ) -> None:
        """
        Args:
            directory: Root the checkpoints are written under, created if missing.
            items: Names of the pytrees written each step, fixed for the run. Each gets its own
                subdirectory, so ``load`` can read one back without materialising the rest.
            config: Run config filed once as checkpoint metadata. Must be JSON-serialisable.
            save_every: Steps between saves. A save also happens whenever preemption is detected.
            keep_latest: How many of the most recent checkpoints survive garbage collection.
            keep_best: How many top-ranked checkpoints survive on top of ``keep_latest``.
            best_metric: Key into the metrics passed to ``save`` that ranks checkpoints.
            best_mode: Whether a high ("max") or low ("min") ``best_metric`` ranks better.
        """
        if best_mode not in ("max", "min"):
            raise ValueError(f"Unrecognised {best_mode=}. Expected 'max' or 'min'")
        if keep_best and best_metric is None:
            raise ValueError("keep_best needs a best_metric to rank checkpoints by")

        self.items = tuple(items)
        self._item_names = frozenset(self.items)
        best_fn = (lambda metrics: metrics[best_metric]) if best_metric else None
        self._ranked = best_fn is not None

        # Latest is always kept so a resume can find the step it left off at.
        preserve: list[ocp.checkpoint_managers.PreservationPolicy] = [
            ocp.checkpoint_managers.LatestN(n=max(keep_latest, 1))
        ]
        if keep_best and best_fn is not None:
            preserve.append(
                ocp.checkpoint_managers.BestN(
                    get_metric_fn=best_fn,
                    n=keep_best,
                    reverse=best_mode == "min",
                    keep_checkpoints_without_metrics=False,
                )
            )

        options = ocp.CheckpointManagerOptions(
            best_fn=best_fn,
            best_mode=best_mode,
            preservation_policy=ocp.checkpoint_managers.AnyPreservationPolicy(preserve),
            save_decision_policy=ocp.checkpoint_managers.AnySavePolicy(
                [
                    # Not sure if we need this as this may exist in the learner before
                    # transferring to the checkpointer as it will avoid the network cost
                    ocp.checkpoint_managers.FixedIntervalPolicy(save_every),
                    ocp.checkpoint_managers.PreemptionCheckpointingPolicy(),
                ]
            ),
        )
        self._manager = ocp.CheckpointManager(
            epath.Path(directory).resolve(),
            options=options,
            handler_registry=_handler_registry(self.items),
            metadata=config,
        )

    def save(
        self,
        step: int,
        state: State,
        metrics: Metrics | None = None,
        *,
        metadata: Config | None = None,
        force: bool = False,
    ) -> bool:
        """Write ``state`` at ``step``, subject to ``save_every``.

        Args:
            step: Step to file the checkpoint under. Steps at or below the latest are skipped.
            state: One pytree per name in ``items`` — all of them, or this raises.
            metrics: Stored with the checkpoint and ranked by ``best_metric``. Each value is
                reduced to a single float, so arrays are meaned.
            metadata: Small JSON-serialisable bookkeeping — step counters and the like. Rides in
                the checkpoint's own metadata file, so reading it back restores no arrays.
            force: Save regardless of ``save_every``, for a run's final checkpoint.

        Returns:
            Whether a checkpoint was written.

        Raises:
            ValueError: If ``state`` does not carry exactly the item names given at construction.
        """
        if state.keys() != self._item_names:
            raise ValueError(f"Expected items {self.items}, got {tuple(state)}")
        # Ask before packing: metrics are device arrays, and reading them stalls on a transfer.
        if not force and not self._manager.should_save(step):
            return False
        args = ocp.args.Composite(**{item: ocp.args.PyTreeSave(tree) for item, tree in state.items()})
        return self._manager.save(
            step,
            args=args,
            metrics=scalars(metrics) if metrics else None,
            custom_metadata=dict(metadata) if metadata else None,
            force=force,
        )

    def metadata(self, step: int | None = None) -> Config | None:
        """Bookkeeping filed with the checkpoint at ``step``, latest when None.

        Args:
            step: Step to read, defaulting to the latest saved.

        Returns:
            What ``save`` filed as ``metadata``, empty when it filed none, and None when there is
            no checkpoint to read.
        """
        step = self.latest if step is None else step
        if step is None:
            return None
        return self._manager.metadata(step).custom_metadata

    def load(self, step: int | None = None, abstract: State | None = None) -> State:
        """Read back a checkpoint.

        Args:
            step: Step to read, defaulting to the latest saved.
            abstract: Pytrees of ``jax.ShapeDtypeStruct`` keyed by item, naming the shardings to
                restore onto. Also selects which items are read, so pass one item to leave the
                rest on disk. Without it orbax reads shardings from the checkpoint, which is
                slower and assumes the topology that saved it.

        Returns:
            One pytree per item read, keyed by item.

        Raises:
            ValueError: If no ``step`` is given and the directory holds no checkpoints.
        """
        step = self.latest if step is None else step
        if step is None:
            raise ValueError(f"No checkpoint to load from {self.directory}")
        if abstract is None:
            args = ocp.args.Composite(**{item: ocp.args.PyTreeRestore() for item in self.items})
        else:
            args = ocp.args.Composite(**{item: ocp.args.PyTreeRestore(tree) for item, tree in abstract.items()})
        return dict(self._manager.restore(step, args=args))

    @property
    def directory(self) -> epath.Path:
        return self._manager.directory

    @property
    def latest(self) -> int | None:
        return self._manager.latest_step()

    @property
    def best(self) -> int | None:
        """None when nothing ranks the checkpoints — orbax would hand back the latest instead."""
        return self._manager.best_step() if self._ranked else None

    @property
    def config(self) -> Config | None:
        return self._manager.metadata().custom_metadata

    def wait(self) -> None:
        self._manager.wait_until_finished()

    def close(self) -> None:
        self._manager.close()

    def __enter__(self) -> "OrbaxCheckpointer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def get_save_path(path: epath.PathLike | None = None, now: datetime | None = None) -> epath.Path:
    """Timestamped directory under ``checkpoints/`` when no path is set, otherwise ``path`` as given."""
    if path:
        return epath.Path(path)
    return epath.Path("checkpoints") / (now or datetime.now(UTC)).strftime("%Y%m%d%H%M%S")
