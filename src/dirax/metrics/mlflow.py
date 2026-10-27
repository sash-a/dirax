import time
from collections.abc import Mapping

from etils import epath
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

from dirax.metrics.scalar import scalars
from dirax.types import Metrics

BATCH_LIMIT = 1000  # mlflow rejects a log_batch carrying more metrics than this.


class MLflowLogger:
    """Logs to one run through a client, so nothing rides on mlflow's process-global active run."""

    def __init__(
        self,
        experiment: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        self.client = MlflowClient(tracking_uri=tracking_uri)
        found = self.client.get_experiment_by_name(experiment)
        experiment_id = found.experiment_id if found else self.client.create_experiment(experiment)
        self.run_id = self.client.create_run(experiment_id, tags=dict(tags or {}), run_name=run_name).info.run_id
        self.closed = False

    def log(self, metrics: Metrics, step: int) -> None:
        timestamp = int(time.time() * 1000)
        entries = [Metric(key, value, timestamp, step) for key, value in scalars(metrics).items()]
        for start in range(0, len(entries), BATCH_LIMIT):
            self.client.log_batch(self.run_id, metrics=entries[start : start + BATCH_LIMIT])

    def log_artifact(self, path: epath.Path, directory: str | None = None) -> None:
        """Uploads from local disk, so a remote ``path`` is mlflow's to reject."""
        if path.is_dir():
            self.client.log_artifacts(self.run_id, str(path), directory)
        else:
            self.client.log_artifact(self.run_id, str(path), directory)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.client.set_terminated(self.run_id)
