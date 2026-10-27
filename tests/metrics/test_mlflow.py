"""The mlflow backend against a real tracking store.

A sqlite store is the cheapest thing mlflow still accepts, and it exercises what a mocked client
cannot: that the entities we hand it are ones it stores, and that a run comes back terminated.
"""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from etils import epath

pytest.importorskip("mlflow", reason="mlflow is an optional extra")

from dirax.metrics.mlflow import BATCH_LIMIT, MLflowLogger


@pytest.fixture(scope="module")
def tracking_uri(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """One store for the module: creating the schema is the slow part, and runs are cheap."""
    yield f"sqlite:///{tmp_path_factory.mktemp('mlflow')}/tracking.db"


@pytest.fixture(autouse=True)
def artifact_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A sqlite store keeps metrics in the database and artifacts beside the working directory, so
    without this every run of the suite leaves an ``mlruns/`` behind wherever it was invoked."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def logger(tracking_uri: str) -> Iterator[MLflowLogger]:
    logger = MLflowLogger("dirax-tests", tracking_uri=tracking_uri)
    yield logger
    logger.close()


def test_a_record_lands_against_its_step_as_one_value_per_key(logger: MLflowLogger) -> None:
    logger.log({"actor/return": np.array([1.0, 3.0]), "learner/loss": 0.5}, 40)

    history = logger.client.get_metric_history(logger.run_id, "actor/return")
    assert [(point.step, point.value) for point in history] == [(40, 2.0)]
    assert logger.client.get_run(logger.run_id).data.metrics["learner/loss"] == 0.5


def test_a_series_keeps_every_step_it_was_logged_at(logger: MLflowLogger) -> None:
    """The step is a position rather than an increment, so re-logging a key extends the series."""
    for step, loss in ((0, 4.0), (10, 2.0), (20, 1.0)):
        logger.log({"loss": loss}, step)

    history = logger.client.get_metric_history(logger.run_id, "loss")
    assert [(point.step, point.value) for point in history] == [(0, 4.0), (10, 2.0), (20, 1.0)]


def test_an_empty_record_reaches_mlflow_as_nothing(logger: MLflowLogger) -> None:
    """A step where no episode finished should not create an empty series."""
    logger.log({"return": np.array([])}, 7)

    assert logger.client.get_run(logger.run_id).data.metrics == {}


def test_the_run_is_named_and_tagged_as_asked(tracking_uri: str) -> None:
    logger = MLflowLogger("dirax-tests", run_name="ppo-cartpole", tracking_uri=tracking_uri, tags={"arm": "baseline"})
    logger.close()

    run = logger.client.get_run(logger.run_id)
    assert run.info.run_name == "ppo-cartpole"
    assert run.data.tags["arm"] == "baseline"


def test_an_experiment_that_already_exists_is_reused_rather_than_recreated(tracking_uri: str) -> None:
    """Every run of a project belongs beside the others, and mlflow refuses a duplicate name."""
    first = MLflowLogger("dirax-shared", tracking_uri=tracking_uri)
    second = MLflowLogger("dirax-shared", tracking_uri=tracking_uri)
    first.close()
    second.close()

    experiments = {logger.client.get_run(logger.run_id).info.experiment_id for logger in (first, second)}
    assert len(experiments) == 1
    assert first.run_id != second.run_id


def test_closing_terminates_the_run_and_can_happen_twice(tracking_uri: str) -> None:
    """Source chains share one backend, so ``close`` arrives once per chain."""
    logger = MLflowLogger("dirax-tests", tracking_uri=tracking_uri)

    logger.close()
    logger.close()

    assert logger.client.get_run(logger.run_id).info.status == "FINISHED"


def test_a_file_artifact_is_filed_under_its_directory(logger: MLflowLogger, tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}")

    logger.log_artifact(epath.Path(config), "actor")

    assert [item.path for item in logger.client.list_artifacts(logger.run_id, "actor")] == ["actor/config.json"]


def test_a_directory_artifact_is_filed_as_its_contents(logger: MLflowLogger, tmp_path: Path) -> None:
    """A checkpoint is a directory, and mlflow needs telling which of its two calls that is."""
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "params").write_text("weights")

    logger.log_artifact(epath.Path(checkpoint), "checkpoints")

    assert [item.path for item in logger.client.list_artifacts(logger.run_id, "checkpoints")] == ["checkpoints/params"]


class BatchRecorder:
    """Stands in for the whole client, to count what one record was split into."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.batches: list[int] = []

    def get_experiment_by_name(self, name: str) -> None:
        return None

    def create_experiment(self, name: str) -> str:
        return "1"

    def create_run(self, experiment_id: str, tags: dict[str, str], run_name: str | None) -> SimpleNamespace:
        return SimpleNamespace(info=SimpleNamespace(run_id="run"))

    def log_batch(self, run_id: str, metrics: list[Any]) -> None:
        self.batches.append(len(metrics))


@pytest.fixture
def batches(monkeypatch: pytest.MonkeyPatch) -> BatchRecorder:
    """Batching is arithmetic on one record, so it is read off the client rather than out of a store."""
    recorder = BatchRecorder()
    monkeypatch.setattr("dirax.metrics.mlflow.MlflowClient", lambda *_, **__: recorder)
    return recorder


def test_a_record_wider_than_mlflow_accepts_is_split_into_batches_it_does(batches: BatchRecorder) -> None:
    """Per-layer gradient norms alone can run to thousands of keys, and mlflow rejects the whole
    batch rather than truncating it, so one wide record would lose every metric in it."""
    logger = MLflowLogger("dirax-tests")

    logger.log({f"grad/{index}": float(index) for index in range(2 * BATCH_LIMIT + 1)}, 0)

    assert batches.batches == [BATCH_LIMIT, BATCH_LIMIT, 1]


def test_an_empty_record_is_not_sent_as_an_empty_batch(batches: BatchRecorder) -> None:
    """mlflow counts an empty batch against the run's request quota and stores nothing for it."""
    logger = MLflowLogger("dirax-tests")

    logger.log({}, 0)

    assert batches.batches == []
