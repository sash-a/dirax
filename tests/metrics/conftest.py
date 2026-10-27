from dataclasses import dataclass, field

import pytest
from etils import epath
from numpy.typing import ArrayLike

from dirax.types import Metrics


@dataclass
class RecordingLogger:
    """A backend that keeps what a chain handed it, so a chain can be read back off its end.

    Stands in for mlflow or wandb: everything a Processor decides — the keys, the step, the
    directory an Artifact is filed under, how many times it reached ``close`` — is visible here
    and nowhere else.
    """

    records: list[tuple[dict[str, ArrayLike], int]] = field(default_factory=list)
    artifacts: list[tuple[epath.Path, str | None]] = field(default_factory=list)
    closes: int = 0

    def log(self, metrics: Metrics, step: int) -> None:
        self.records.append((dict(metrics), step))

    def log_artifact(self, path: epath.Path, directory: str | None = None) -> None:
        self.artifacts.append((path, directory))

    def close(self) -> None:
        self.closes += 1

    @property
    def keys(self) -> list[list[str]]:
        """The keys of each record in order, which is what namespacing and windowing decide."""
        return [sorted(metrics) for metrics, _ in self.records]

    @property
    def steps(self) -> list[int]:
        return [step for _, step in self.records]


@pytest.fixture
def backend() -> RecordingLogger:
    return RecordingLogger()
