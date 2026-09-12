from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, NamedTuple, Protocol, final

import orbax.checkpoint as ocp
from etils import epath

from brrrl.metrics.scalar import scalars
from brrrl.types import Metrics, UpdateState

type Config = Mapping[str, Any]


class StateLayout[S: UpdateState](Protocol):
    """Methods that transform an `UpdateState` to and from a checkpointable `Mapping`"""

    @property
    def items(self) -> tuple[str, ...]: ...
    def to_items(self, state: S) -> Mapping[str, Any]: ...
    def from_items(self, loaded: Mapping[str, Any]) -> S: ...


@dataclass
class NamedTupleLayout[S: NamedTuple]:
    """Transforms an `UpdateState` backed by a `NamedTuple` to/from a checkpointable `Mapping`.

    Bound on `NamedTuple` rather than `UpdateState`: a NamedTuple is already everything this needs,
    and the checkpointer it is handed to is what insists on the state being an `UpdateState`.
    """

    state_type: type[S]

    @property
    def items(self) -> tuple[str, ...]:
        return self.state_type._fields

    def to_items(self, state: S) -> Mapping[str, Any]:
        return state._asdict()

    def from_items(self, loaded: Mapping[str, Any]) -> S:
        return self.state_type(**loaded)


class Checkpointer[S: UpdateState](Protocol):
    def save(
        self,
        step: int,
        state: S,
        metrics: Metrics | None = None,
        *,
        metadata: Config | None = None,
        force: bool = False,
    ) -> bool:
        """Write ``state`` at ``step``.

        Args:
            step: Step to file the checkpoint under. Steps at or below the latest are skipped.
            state: The run's state, as one pytree.
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

    def load(self, step: int | None = None, abstract: S | None = None) -> S:
        """Read back a checkpoint.

        Args:
            step: Step to read, defaulting to the latest saved.
            abstract: The state's shape as ``jax.ShapeDtypeStruct`` leaves, naming what to restore
                onto. Leaves marked ``ocp.PLACEHOLDER`` are left on disk.

        Returns:
            The state.
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


def _handler_registry(items: tuple[str, ...]) -> ocp.DefaultCheckpointHandlerRegistry:
    handler = ocp.PyTreeCheckpointHandler()
    registry = ocp.DefaultCheckpointHandlerRegistry()
    for item in items:
        registry.add(item, ocp.args.PyTreeSave, handler)
        registry.add(item, ocp.args.PyTreeRestore, handler)
    return registry


@final
class OrbaxCheckpointer[S: UpdateState]:
    """Saves a run's state as orbax checkpoints under one local directory.

    A state goes in and comes out whole; the ``layout`` is what knows it is filed as one item per
    part, so callers never meet orbax's args vocabulary and never name an item themselves.
    """

    def __init__(
        self,
        directory: epath.PathLike,
        layout: StateLayout[S],
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
            layout: `StateLayout` that transforms `UpdateState` to/from checkpointable `Mappings`.
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

        best_fn = (lambda metrics: metrics[best_metric]) if best_metric else None
        self._ranked = best_fn is not None

        self.layout = layout

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
            handler_registry=_handler_registry(layout.items),
            metadata=config,
        )

    def save(
        self,
        step: int,
        state: S,
        metrics: Metrics | None = None,
        *,
        metadata: Config | None = None,
        force: bool = False,
    ) -> bool:
        """Write ``state`` at ``step``, subject to ``save_every``.

        Args:
            step: Step to file the checkpoint under. Steps at or below the latest are skipped.
            state: The run's state, as one pytree.
            metrics: Stored with the checkpoint and ranked by ``best_metric``. Each value is
                reduced to a single float, so arrays are meaned.
            metadata: Small JSON-serialisable bookkeeping — step counters and the like. Rides in
                the checkpoint's own metadata file, so reading it back restores no arrays.
            force: Save regardless of ``save_every``, for a run's final checkpoint.

        Returns:
            Whether a checkpoint was written.
        """
        # Ask before packing: metrics are device arrays, and reading them stalls on a transfer.
        if not force and not self._manager.should_save(step):
            return False

        checkpointable = {key: ocp.args.PyTreeSave(tree) for key, tree in self.layout.to_items(state).items()}
        args = ocp.args.Composite(**checkpointable)
        return self._manager.save(
            step,
            args=args,
            metrics=scalars(metrics) if metrics else None,
            custom_metadata=dict(metadata) if metadata else None,
            force=force,
        )

    def load(self, step: int | None = None, abstract: S | None = None) -> S:
        """Read back a checkpoint.

        Args:
            step: Step to read, defaulting to the latest saved.
            abstract: A state of ``jax.ShapeDtypeStruct`` leaves.
        Returns:
            The state, rebuilt by the layout from the items on disk.

        Raises:
            ValueError: If no ``step`` is given and the directory holds no checkpoints.
        """
        step = self.latest if step is None else step
        if step is None:
            raise ValueError(f"No checkpoint to load from {self.directory}")
        # Every item is asked for, so the layout always has the whole set to rebuild from.
        trees = self.layout.to_items(abstract) if abstract is not None else dict.fromkeys(self.layout.items)
        args = ocp.args.Composite(**{item: ocp.args.PyTreeRestore(tree) for item, tree in trees.items()})
        return self.layout.from_items(dict(self._manager.restore(step, args=args)))

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

    def __enter__(self) -> "OrbaxCheckpointer[S]":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def get_save_path(path: epath.PathLike | None = None, now: datetime | None = None) -> epath.Path:
    """Timestamped directory under ``checkpoints/`` when no path is set, otherwise ``path`` as given."""
    if path:
        return epath.Path(path)
    return epath.Path("checkpoints") / (now or datetime.now(UTC)).strftime("%Y%m%d%H%M%S")
