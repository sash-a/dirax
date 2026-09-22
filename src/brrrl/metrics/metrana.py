from collections.abc import Mapping, Sequence
from typing import Any

import metrana
from etils import epath

from brrrl.metrics.scalar import scalars
from brrrl.types import Metrics


class MetranaLogger:
    """Drives metrana's process-global run, so a second logger in the process is the same run.

    `experiment` is metrana's grouping over runs; unset leaves the run ungrouped, which the UI
    then files under whatever it infers.
    """

    def __init__(
        self,
        workspace: str,
        project: str,
        run_name: str,
        experiment: str | None = None,
        config: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        description: str | None = None,
    ) -> None:
        metrana.init(
            workspace_name=workspace,
            project_name=project,
            run_name=run_name,
            experiment_name=experiment,
            config=dict(config) if config else None,
            tags=list(tags) if tags else None,
            description=description,
        )
        self.closed = False

    def log(self, metrics: Metrics, step: int) -> None:
        """Merge semantics, because the step is a clock rather than a count of these calls.

        A Learner's window advances it by nothing, so two of them land on the step an Actor last
        left, and ordered writes take a repeat as a series going backwards.
        """
        values = scalars(metrics)
        if values:
            metrana.log_distributed(values, step=step)

    def log_artifact(self, path: epath.Path, directory: str | None = None) -> None:
        """The directory it is filed under is the artifact's type, and versions accrue under it.

        A nested directory folds into one segment with dashes, so the type stays a single name.
        """
        metrana.upload_artifact(
            str(path), type=directory.replace("/", "-") if directory else "artifact", name=path.name
        )

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            metrana.close()
