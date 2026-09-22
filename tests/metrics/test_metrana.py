"""The metrana backend against a module that records what it was handed.

metrana's transport is not ours to test, and it holds no run object to inspect: ``init`` configures
the module and every call after it is a module-level function. What is ours is which of those we
reach for — that a record goes out under merge semantics rather than ordered ones, that values are
reduced first, and that an artifact's type is the directory it was filed under.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from etils import epath

pytest.importorskip("metrana", reason="metrana is an optional extra")

from brrrl.metrics.metrana import MetranaLogger


@dataclass
class FakeMetrana:
    """Stands in for the metrana module, plus the arguments its run was started with."""

    starts: list[dict[str, Any]] = field(default_factory=list)
    ordered: list[tuple[dict[str, float], int]] = field(default_factory=list)
    merged: list[tuple[dict[str, float], int]] = field(default_factory=list)
    artifacts: list[tuple[str, str, str]] = field(default_factory=list)
    closes: int = 0

    def init(self, **arguments: Any) -> None:
        self.starts.append(arguments)

    def log(self, values: dict[str, float], step: int) -> None:
        self.ordered.append((values, step))

    def log_distributed(self, values: dict[str, float], step: int) -> None:
        self.merged.append((values, step))

    def upload_artifact(self, files: str, *, type: str, name: str) -> None:
        self.artifacts.append((files, type, name))

    def close(self) -> None:
        self.closes += 1


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> FakeMetrana:
    """Replaces the functions the backend reaches for, so nothing here opens a run or a disk spool."""
    fake = FakeMetrana()
    for name in ("init", "log", "log_distributed", "upload_artifact", "close"):
        monkeypatch.setattr(f"brrrl.metrics.metrana.metrana.{name}", getattr(fake, name))
    return fake


def test_a_run_is_started_with_the_identity_it_was_given(module: FakeMetrana) -> None:
    MetranaLogger(
        "team",
        "brrrl",
        "ppo-cartpole",
        experiment="ladder",
        config={"lr": 0.1},
        tags=("baseline", "gpu"),
        description="a ladder rung",
    )

    assert module.starts == [
        {
            "workspace_name": "team",
            "project_name": "brrrl",
            "run_name": "ppo-cartpole",
            "experiment_name": "ladder",
            "config": {"lr": 0.1},
            "tags": ["baseline", "gpu"],
            "description": "a ladder rung",
        }
    ]


def test_what_was_not_given_is_left_unset_rather_than_sent_empty(module: FakeMetrana) -> None:
    """An empty config or tag list is a claim about the run; metrana's own default is not."""
    MetranaLogger("team", "brrrl", "ppo-cartpole")

    assert module.starts[0]["experiment_name"] is None
    assert module.starts[0]["config"] is None
    assert module.starts[0]["tags"] is None


def test_a_record_is_reduced_and_logged_against_its_step(module: FakeMetrana) -> None:
    MetranaLogger("team", "brrrl", "run").log({"return": np.array([1.0, 3.0]), "loss": 0.5}, 40)

    assert module.merged == [({"return": 2.0, "loss": 0.5}, 40)]


def test_records_go_out_under_merge_semantics_and_never_ordered_ones(module: FakeMetrana) -> None:
    """The step is a clock, not a count of these calls: a Learner's window advances it by nothing, so
    two of them land on the step an Actor last left. Ordered writes take that repeat as a series
    going backwards, and a metric's first point fixes which semantics it keeps for the whole run."""
    logger = MetranaLogger("team", "brrrl", "run")

    logger.log({"loss": 1.0}, 100)
    logger.log({"loss": 0.5}, 100)

    assert module.merged == [({"loss": 1.0}, 100), ({"loss": 0.5}, 100)]
    assert module.ordered == []


def test_an_empty_record_is_not_logged(module: FakeMetrana) -> None:
    """A log with no values is still a point on the run's timeline, and the step is not this backend's to move."""
    MetranaLogger("team", "brrrl", "run").log({"return": np.array([])}, 7)

    assert module.merged == []


def test_an_artifact_is_typed_by_the_directory_it_was_filed_under(module: FakeMetrana) -> None:
    """Versions accrue under (type, name), so the place has to be the type for two Origins'
    checkpoints to stay apart rather than version over each other."""
    MetranaLogger("team", "brrrl", "run").log_artifact(epath.Path("/run/ckpt"), "learner/checkpoints")

    assert module.artifacts == [("/run/ckpt", "learner-checkpoints", "ckpt")]


def test_an_artifact_filed_under_no_directory_takes_a_type_of_its_own(module: FakeMetrana) -> None:
    MetranaLogger("team", "brrrl", "run").log_artifact(epath.Path("/run/video.mp4"))

    assert module.artifacts == [("/run/video.mp4", "artifact", "video.mp4")]


def test_the_run_is_closed_once_however_many_chains_close_it(module: FakeMetrana) -> None:
    """Closing twice raises MetranaNotInitializedError, and every chain in a run shares this one backend."""
    logger = MetranaLogger("team", "brrrl", "run")

    logger.close()
    logger.close()

    assert module.closes == 1
