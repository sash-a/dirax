"""The wandb backend against a run that records what it was handed.

wandb's own transport is not ours to test, and offline mode still spawns a service and writes a
run directory. What is ours is the call we make into it: which arguments a run is started with,
that values are reduced before they go in, and that an Artifact's name survives having no slashes.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from etils import epath

pytest.importorskip("wandb", reason="wandb is an optional extra")

from brrrl.metrics.wandb import WandbLogger


@dataclass
class FakeRun:
    """Stands in for the wandb run a logger holds, plus the arguments it was started with."""

    starts: list[dict[str, Any]] = field(default_factory=list)
    records: list[tuple[dict[str, float], int]] = field(default_factory=list)
    artifacts: list[tuple[str, str, str]] = field(default_factory=list)
    finishes: int = 0

    def log(self, values: dict[str, float], step: int) -> None:
        self.records.append((values, step))

    def log_artifact(self, path: str, name: str, type: str) -> None:
        self.artifacts.append((path, name, type))

    def finish(self) -> None:
        self.finishes += 1


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    """Replaces ``wandb.init``, so nothing here reaches a wandb directory or a service process."""
    run = FakeRun()

    def init(**arguments: Any) -> FakeRun:
        run.starts.append(arguments)
        return run

    monkeypatch.setattr("brrrl.metrics.wandb.wandb.init", init)
    return run


def test_a_run_is_started_with_the_identity_it_was_given(run: FakeRun) -> None:
    WandbLogger(
        "brrrl", name="ppo-cartpole", entity="team", group="ladder", config={"lr": 0.1}, tags=("baseline", "gpu")
    )

    assert run.starts == [
        {
            "project": "brrrl",
            "name": "ppo-cartpole",
            "entity": "team",
            "group": "ladder",
            "config": {"lr": 0.1},
            "tags": ["baseline", "gpu"],
        }
    ]


def test_what_was_not_given_is_left_unset_rather_than_sent_empty(run: FakeRun) -> None:
    """An empty config or tag list is a claim about the run; wandb's own default is not."""
    WandbLogger("brrrl")

    assert run.starts[0]["config"] is None
    assert run.starts[0]["tags"] is None


def test_a_record_is_reduced_and_logged_against_its_step(run: FakeRun) -> None:
    WandbLogger("brrrl").log({"return": np.array([1.0, 3.0]), "loss": 0.5}, 40)

    assert run.records == [({"return": 2.0, "loss": 0.5}, 40)]


def test_an_empty_record_is_not_logged(run: FakeRun) -> None:
    """wandb takes a log with no values as a step of its own, and the step is not this backend's to move."""
    WandbLogger("brrrl").log({"return": np.array([])}, 7)

    assert run.records == []


def test_an_artifacts_directory_folds_into_a_name_with_no_slashes(run: FakeRun) -> None:
    """wandb rejects a slash in an artifact name, so the place it was filed under has to survive
    as part of the name or two Origins' checkpoints collide."""
    WandbLogger("brrrl").log_artifact(epath.Path("/run/ckpt"), "learner/checkpoints")

    assert run.artifacts == [("/run/ckpt", "learner-checkpoints-ckpt", "artifact")]


def test_an_artifact_filed_under_no_directory_is_named_by_itself(run: FakeRun) -> None:
    WandbLogger("brrrl").log_artifact(epath.Path("/run/video.mp4"))

    assert run.artifacts == [("/run/video.mp4", "video.mp4", "artifact")]


def test_the_run_is_finished_once_however_many_chains_close_it(run: FakeRun) -> None:
    """Finishing twice is an error in wandb, and every chain in a run shares this one backend."""
    logger = WandbLogger("brrrl")

    logger.close()
    logger.close()

    assert run.finishes == 1
