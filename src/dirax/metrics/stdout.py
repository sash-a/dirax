import sys
from dataclasses import dataclass

from etils import epath

from dirax.metrics.scalar import scalars
from dirax.types import Metrics


@dataclass
class StdoutLogger:
    """Prints one line per record. Enough to watch a run without a tracking server."""

    precision: int = 4

    def log(self, metrics: Metrics, step: int) -> None:
        values = scalars(metrics)
        if not values:
            return
        body = " ".join(f"{key}={value:.{self.precision}g}" for key, value in sorted(values.items()))
        print(f"[{step}] {body}", flush=True)

    def log_artifact(self, path: epath.Path, directory: str | None = None) -> None:
        destination = f"{directory}/{path.name}" if directory else path.name
        print(f"[artifact] {destination} <- {path}", flush=True)

    def close(self) -> None:
        sys.stdout.flush()
