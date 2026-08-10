from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

from echo import TcpClient

from brrrl.acting.roller import Roller
from brrrl.distributed.placement.component import Behaviour
from brrrl.metrics.collector import MetricRecorder


@dataclass
class Actor:
    """One thread per roller. Rollers are built here, not shipped here."""

    RAY_OPTIONS: ClassVar[Behaviour] = {}

    build_rollers: Callable[[], Sequence[Roller]]
    build_client: Callable[[], TcpClient]
    metrics: MetricRecorder

    def run(self) -> None: ...
