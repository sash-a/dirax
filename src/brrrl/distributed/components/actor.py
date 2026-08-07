from collections.abc import Callable, Sequence
from dataclasses import dataclass

from echo import TcpClient

from brrrl.acting.roller import Roller
from brrrl.logging.logger import Logger


@dataclass
class Actor:
    """One thread per roller. Rollers are built here, not shipped here."""

    build_rollers: Callable[[], Sequence[Roller]]
    build_client: Callable[[], TcpClient]
    logger: Logger

    def run(self) -> None: ...
