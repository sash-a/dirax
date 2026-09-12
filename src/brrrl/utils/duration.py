import time
from dataclasses import dataclass, field
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


@dataclass
class Schedule:
    """A stateful version of `Duration`."""

    every: Duration
    prev: Instant | None = field(init=False, default=None)
    ticks: int = field(init=False, default=0)

    def due(self) -> bool:
        self.ticks += 1
        now = Instant.now(self.ticks)
        if self.prev is None or self.every.due(now, self.prev):
            self.prev = now
            return True
        else:
            return False


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
