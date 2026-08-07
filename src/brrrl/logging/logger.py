from typing import Protocol

from brrrl.types import Metrics


class Logger(Protocol):
    def log(self, metrics: Metrics, step: int) -> None: ...
