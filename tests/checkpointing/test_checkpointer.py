"""The checkpointer against real orbax on a real directory.

A checkpoint is only worth what a resume can read back out of it, and every interesting behaviour
here — the save policy, retention, ranking, what a partial restore materialises — is orbax's to
enforce on disk. Faking the manager would only assert that we call it the way we already call it,
so these write files and read them back, and the arrays stay small enough that it costs
milliseconds.
"""

from datetime import UTC, datetime

import jax
import numpy as np
import pytest
from etils import epath

from brrrl.checkpointing.checkpointer import OrbaxCheckpointer, get_save_path


def params(value: float) -> dict[str, np.ndarray]:
    return {"w": np.full((2,), value, dtype=np.float32)}


def steps_on_disk(directory: epath.Path) -> list[int]:
    return sorted(int(child.name) for child in directory.iterdir() if child.name.isdigit())


def test_a_saved_step_reads_back_as_the_arrays_that_went_in(tmp_path: epath.PathLike) -> None:
    with OrbaxCheckpointer(tmp_path, ["params", "opt"]) as checkpointer:
        checkpointer.save(0, {"params": params(1.5), "opt": {"mu": np.zeros((3,), np.float32)}})
        checkpointer.wait()

        restored = checkpointer.load()

    assert sorted(restored) == ["opt", "params"]
    np.testing.assert_array_equal(restored["params"]["w"], params(1.5)["w"])
    np.testing.assert_array_equal(restored["opt"]["mu"], np.zeros((3,), np.float32))


def test_saves_land_only_on_the_interval(tmp_path: epath.Path) -> None:
    """``save`` reports whether the policy admitted the step, so a caller can log or count it."""
    with OrbaxCheckpointer(tmp_path, ["params"], save_every=5, keep_latest=3) as checkpointer:
        admitted = [checkpointer.save(step, {"params": params(step)}) for step in (0, 3, 5, 9, 10)]
        checkpointer.wait()

        assert admitted == [True, False, True, False, True]
        assert steps_on_disk(checkpointer.directory) == [0, 5, 10]


def test_forcing_lands_a_final_checkpoint_off_the_interval(tmp_path: epath.Path) -> None:
    """A run ends where it ends, which is rarely on a multiple of ``save_every``."""
    with OrbaxCheckpointer(tmp_path, ["params"], save_every=100, keep_latest=2) as checkpointer:
        checkpointer.save(0, {"params": params(0)})
        forced = checkpointer.save(7, {"params": params(7)}, force=True)
        checkpointer.wait()

        assert forced
        assert checkpointer.latest == 7


def test_a_step_at_or_below_the_latest_is_skipped(tmp_path: epath.Path) -> None:
    """Resuming replays the steps already on disk; they must not be written twice."""
    with OrbaxCheckpointer(tmp_path, ["params"]) as checkpointer:
        checkpointer.save(10, {"params": params(10)})
        checkpointer.wait()

        assert not checkpointer.save(10, {"params": params(0)})
        assert not checkpointer.save(4, {"params": params(0)})
        assert checkpointer.latest == 10


@pytest.mark.parametrize(
    "state",
    [
        pytest.param({"params": params(1)}, id="missing"),
        pytest.param({"params": params(1), "opt": params(1), "extra": params(1)}, id="unexpected"),
        pytest.param({"params": params(1), "optimiser": params(1)}, id="misspelled"),
    ],
)
def test_state_that_is_not_the_declared_items_is_refused_before_anything_is_written(
    tmp_path: epath.Path, state: dict[str, dict[str, np.ndarray]]
) -> None:
    """Items are fixed for the run, so a state that does not match them is a wiring mistake. Failing
    on the first save beats a resume discovering months later that an item was never written."""
    with OrbaxCheckpointer(tmp_path, ["params", "opt"]) as checkpointer:
        with pytest.raises(ValueError, match="params"):
            checkpointer.save(0, state, force=True)

        assert checkpointer.latest is None


def test_only_the_requested_items_are_read_back(tmp_path: epath.Path) -> None:
    """The point of an item per pytree: an evaluator wants the params and not the optimiser state."""
    with OrbaxCheckpointer(tmp_path, ["params", "opt"]) as checkpointer:
        checkpointer.save(0, {"params": params(2), "opt": {"mu": np.ones((512,), np.float32)}})
        checkpointer.wait()

        restored = checkpointer.load(abstract={"params": {"w": jax.ShapeDtypeStruct((2,), np.float32)}})

    assert list(restored) == ["params"]
    np.testing.assert_array_equal(restored["params"]["w"], params(2)["w"])


def test_restoring_onto_shape_dtype_structs_gives_arrays_of_that_shape_and_dtype(tmp_path: epath.Path) -> None:
    """What a resume actually passes: the shapes it already knows, so orbax reads no topology off disk."""
    with OrbaxCheckpointer(tmp_path, ["params"]) as checkpointer:
        checkpointer.save(0, {"params": params(3)})
        checkpointer.wait()

        restored = checkpointer.load(0, {"params": {"w": jax.ShapeDtypeStruct((2,), np.float32)}})

    assert restored["params"]["w"].shape == (2,)
    assert restored["params"]["w"].dtype == np.float32


def test_metadata_filed_with_a_checkpoint_reads_back_without_restoring_it(tmp_path: epath.Path) -> None:
    """Step counters ride beside the arrays so a resume can read them without materialising a pytree."""
    with OrbaxCheckpointer(tmp_path, ["params"]) as checkpointer:
        checkpointer.save(4, {"params": params(4)}, metadata={"consumed": 128, "version": 3})
        checkpointer.wait()

        assert checkpointer.metadata() == {"consumed": 128, "version": 3}
        assert checkpointer.metadata(4) == {"consumed": 128, "version": 3}


def test_a_checkpoint_that_filed_no_metadata_reads_back_empty_rather_than_missing(tmp_path: epath.Path) -> None:
    """None means "no checkpoint"; a checkpoint that simply filed nothing is a different answer."""
    with OrbaxCheckpointer(tmp_path, ["params"]) as checkpointer:
        checkpointer.save(0, {"params": params(0)})
        checkpointer.wait()

        assert checkpointer.metadata() == {}


def test_the_run_config_is_filed_once_for_the_directory_rather_than_per_step(tmp_path: epath.Path) -> None:
    with OrbaxCheckpointer(tmp_path, ["params"], config={"lr": 0.1, "seed": 7}) as checkpointer:
        checkpointer.save(0, {"params": params(0)})
        checkpointer.wait()

        assert checkpointer.config == {"lr": 0.1, "seed": 7}
        assert checkpointer.metadata() == {}


def test_an_empty_directory_has_no_latest_and_nothing_to_load(tmp_path: epath.Path) -> None:
    """The state a run starts in when nobody has resumed anything, and it must not look like a checkpoint."""
    with OrbaxCheckpointer(tmp_path, ["params"]) as checkpointer:
        assert checkpointer.latest is None
        assert checkpointer.metadata() is None

        with pytest.raises(ValueError, match="No checkpoint"):
            checkpointer.load()


def test_the_directory_is_where_the_checkpoints_landed(tmp_path: epath.Path) -> None:
    """It is handed to a metric chain as an Artifact path, so it names the resolved root, not a step."""
    with OrbaxCheckpointer(tmp_path / "run", ["params"]) as checkpointer:
        checkpointer.save(0, {"params": params(0)})
        checkpointer.wait()

        assert checkpointer.directory == epath.Path(tmp_path / "run").resolve()
        assert (checkpointer.directory / "0").exists()


def test_only_the_most_recent_checkpoints_survive(tmp_path: epath.Path) -> None:
    """Retention is what keeps a long run's disk bounded, and it happens as saves land."""
    with OrbaxCheckpointer(tmp_path, ["params"], keep_latest=2) as checkpointer:
        for step in range(5):
            checkpointer.save(step, {"params": params(step)})
        checkpointer.wait()

        assert steps_on_disk(checkpointer.directory) == [3, 4]


def test_the_latest_checkpoint_survives_even_when_none_were_asked_for(tmp_path: epath.Path) -> None:
    """Keeping nothing would leave a run unable to resume, so ``keep_latest=0`` still keeps one."""
    with OrbaxCheckpointer(tmp_path, ["params"], keep_latest=0) as checkpointer:
        for step in range(3):
            checkpointer.save(step, {"params": params(step)})
        checkpointer.wait()

        assert steps_on_disk(checkpointer.directory) == [2]
        assert checkpointer.latest == 2


def test_the_best_ranked_checkpoint_survives_alongside_the_latest(tmp_path: epath.Path) -> None:
    """The two reasons to keep a checkpoint are independent: resume from the latest, ship the best."""
    checkpointer = OrbaxCheckpointer(
        tmp_path, ["params"], keep_latest=1, keep_best=1, best_metric="return", best_mode="max"
    )
    with checkpointer:
        for step, episode_return in ((0, 1.0), (1, 5.0), (2, 2.0), (3, 4.0)):
            checkpointer.save(step, {"params": params(step)}, {"return": np.float32(episode_return)})
        checkpointer.wait()

        assert checkpointer.best == 1
        assert checkpointer.latest == 3
        assert steps_on_disk(checkpointer.directory) == [1, 3]


def test_the_best_step_is_the_lowest_one_when_the_mode_is_min(tmp_path: epath.Path) -> None:
    checkpointer = OrbaxCheckpointer(tmp_path, ["params"], keep_best=1, best_metric="loss", best_mode="min")
    with checkpointer:
        for step, loss in ((0, 3.0), (1, 0.5), (2, 9.0)):
            checkpointer.save(step, {"params": params(step)}, {"loss": loss})
        checkpointer.wait()

        assert checkpointer.best == 1


def test_nothing_is_ranked_without_a_metric_to_rank_by(tmp_path: epath.Path) -> None:
    """Orbax hands back the latest step when no ``best_fn`` was configured, which would read as a
    ranking that had happened. Unranked has to be visibly unranked."""
    with OrbaxCheckpointer(tmp_path, ["params"]) as checkpointer:
        checkpointer.save(0, {"params": params(0)}, {"return": 1.0})
        checkpointer.wait()

        assert checkpointer.latest == 0
        assert checkpointer.best is None


def test_array_metrics_are_reduced_to_one_float_before_they_rank_anything(tmp_path: epath.Path) -> None:
    """Metrics arrive per env, so ranking has to compare summaries: step 0 wins on the mean while
    losing on the maximum, and a checkpointer that ranked elementwise would keep the other one."""
    checkpointer = OrbaxCheckpointer(tmp_path, ["params"], keep_best=1, best_metric="return", best_mode="max")
    with checkpointer:
        checkpointer.save(0, {"params": params(0)}, {"return": np.array([5.0, 5.0], np.float32)})
        checkpointer.save(1, {"params": params(1)}, {"return": np.array([9.0, 0.0], np.float32)})
        checkpointer.wait()

        assert checkpointer.best == 0


def test_an_unrecognised_best_mode_is_refused_at_construction(tmp_path: epath.Path) -> None:
    with pytest.raises(ValueError, match="best_mode"):
        OrbaxCheckpointer(tmp_path, ["params"], keep_best=1, best_metric="return", best_mode="highest")


def test_keeping_best_checkpoints_needs_a_metric_to_rank_them_by(tmp_path: epath.Path) -> None:
    """Otherwise the request is silently dropped and a run finishes holding none of what it asked for."""
    with pytest.raises(ValueError, match="best_metric"):
        OrbaxCheckpointer(tmp_path, ["params"], keep_best=2)


def test_close_is_idempotent(tmp_path: epath.Path) -> None:
    """A run reaches it from its shutdown path and again from a context manager unwinding."""
    checkpointer = OrbaxCheckpointer(tmp_path, ["params"])
    checkpointer.save(0, {"params": params(0)})

    checkpointer.close()
    checkpointer.close()


def test_leaving_the_context_manager_lands_what_a_fresh_checkpointer_reads_back(tmp_path: epath.Path) -> None:
    """Saves are asynchronous, so the block exiting is what makes them another process's business."""
    with OrbaxCheckpointer(tmp_path, ["params"]) as checkpointer:
        checkpointer.save(6, {"params": params(6)}, metadata={"consumed": 12})

    with OrbaxCheckpointer(tmp_path, ["params"]) as resumed:
        assert resumed.latest == 6
        assert resumed.metadata() == {"consumed": 12}
        np.testing.assert_array_equal(resumed.load()["params"]["w"], params(6)["w"])


def test_an_unset_save_path_is_a_timestamped_directory_under_checkpoints() -> None:
    """Two runs started from the same config must not write into each other."""
    stamped = get_save_path(now=datetime(2026, 8, 10, 22, 4, 5, tzinfo=UTC))

    assert stamped == epath.Path("checkpoints/20260810220405")


@pytest.mark.parametrize("unset", [None, ""], ids=["none", "empty"])
def test_the_timestamp_is_taken_now_when_no_clock_is_given(unset: str | None) -> None:
    stamped = get_save_path(unset)

    assert stamped.parent == epath.Path("checkpoints")
    assert stamped.name.isdigit() and len(stamped.name) == len("20260810220405")


def test_a_path_that_was_set_is_used_exactly_as_given(tmp_path: epath.Path) -> None:
    """A resume is pointed at the directory it left off in, and must not land in a new one."""
    assert get_save_path(tmp_path / "run-17") == epath.Path(tmp_path / "run-17")
