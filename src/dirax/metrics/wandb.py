from collections.abc import Mapping, Sequence
from typing import Any

from etils import epath

import wandb
from dirax.metrics.scalar import scalars
from dirax.types import Metrics


class WandbLogger:
    """Logs to one run held on the instance, so a second logger in the process is a second run."""

    def __init__(
        self,
        project: str,
        name: str | None = None,
        entity: str | None = None,
        group: str | None = None,
        config: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> None:
        self.run = wandb.init(
            project=project,
            name=name,
            entity=entity,
            group=group,
            config=dict(config) if config else None,
            tags=list(tags) if tags else None,
        )
        self.closed = False

    def log(self, metrics: Metrics, step: int) -> None:
        values = scalars(metrics)
        if values:
            self.run.log(values, step=step)

    def log_artifact(self, path: epath.Path, directory: str | None = None) -> None:
        destination = f"{directory}/{path.name}" if directory else path.name
        # Artifact names take no slashes, so the directory folds into the name with dashes.
        self.run.log_artifact(str(path), name=destination.replace("/", "-"), type="artifact")

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.run.finish()
