import time
from typing import NamedTuple, Protocol, Self


class Instant(NamedTuple):
    step: int
    time: float

    @classmethod
    def now(cls, step: int) -> Self:
        return cls(step, time.monotonic())


class Duration(Protocol):
    """How much has to pass between acts, counted in whichever of an `Instant`'s clocks it reads."""

    def due(self, now: Instant, prev: Instant) -> bool:
        """Whether ``now`` is a whole duration past ``prev``.

        A question, not a claim: asking twice gives the same answer, and the caller is what records
        having acted by moving the ``prev`` it asks with.
        """
        ...


class EveryTicks:
    """Every ``n`` of whatever the caller counts."""

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"A period spans at least one tick, got {n=}")
        self.n = n

    def due(self, now: Instant, prev: Instant) -> bool:
        return now.step - prev.step >= self.n


class EverySeconds:
    """Every ``seconds`` of wall clock, however many ticks that spans."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def due(self, now: Instant, prev: Instant) -> bool:
        return now.time - prev.time > self.seconds
