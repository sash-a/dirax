import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

from numpy.typing import ArrayLike

from brrrl.metrics.accumulator import MetricAccumulator
from brrrl.types import Metrics


class Elapsed(Protocol):
    """Elapsed time, allowing for adding times (fold) and getting results (dump)"""

    def fold(self, seconds: float) -> None: ...
    def dump(self) -> ArrayLike: ...


@dataclass(slots=True)
class AllElapsed:
    """Every elapsed time, summarisable downstream."""

    seconds: list[float] = field(default_factory=list)

    def fold(self, seconds: float) -> None:
        self.seconds.append(seconds)

    def dump(self) -> ArrayLike:
        return self.seconds


@dataclass(slots=True)
class AggregatedElapsed:
    """The running mean of the elapsed times."""

    accumulator: MetricAccumulator = field(default_factory=MetricAccumulator)

    def fold(self, seconds: float) -> None:
        self.accumulator.fold(seconds)

    def dump(self) -> ArrayLike:
        return self.accumulator.mean


@dataclass
class Timer:
    """Records how long named blocks took, and dumps them as metrics.

        timer = Timer()
        with timer("rollout_time"):
            ...
        recorder.record(timer.dump())

    Timer(AllElapsed) keeps every sample for downstream aggregation.
    Timer(AggregatedElapsed) keeps a running mean instead.

    JAX dispatches asynchronously, so a block that only launches work times the launch. Block on
    what it produced before leaving the block.
    """

    elapsed: Callable[[], Elapsed] = AllElapsed
    times: dict[str, Elapsed] = field(default_factory=dict)

    @contextmanager
    def __call__(self, key: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.times.setdefault(key, self.elapsed()).fold(time.perf_counter() - start)

    def dump(self) -> Metrics:
        """Everything timed since the last dump, which is then forgotten: the window has moved on."""
        times, self.times = self.times, {}
        return {key: elapsed.dump() for key, elapsed in times.items()}
