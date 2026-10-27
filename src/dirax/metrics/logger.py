from dataclasses import dataclass
from typing import Protocol

from etils import epath

from dirax.types import Metrics


class Logger(Protocol):
    def log(self, metrics: Metrics, step: int) -> None: ...

    def log_artifact(self, path: epath.Path, directory: str | None = None) -> None:
        """Store the file or directory at ``path`` under ``directory`` in the run."""
        ...

    def close(self) -> None:
        """Flush and release. Must be idempotent: chains share a backend."""
        ...


@dataclass
class MultiLogger:
    """Holds multiple loggers and logs to them all."""

    loggers: list[Logger]

    def log(self, metrics: Metrics, step: int) -> None:
        for logger in self.loggers:
            logger.log(metrics, step)

    def log_artifact(self, path: epath.Path, directory: str | None = None) -> None:
        for logger in self.loggers:
            logger.log_artifact(path, directory)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()


@dataclass
class NamespacedProcessor:
    prefix: str
    logger: Logger

    def log(self, metrics: Metrics, step: int) -> None:
        self.logger.log({f"{self.prefix}/{key}": value for key, value in metrics.items()}, step)

    def log_artifact(self, path: epath.Path, directory: str | None = None) -> None:
        self.logger.log_artifact(path, self.prefix if directory is None else f"{self.prefix}/{directory}")

    def close(self) -> None:
        self.logger.close()
