"""What a Timer promises the metrics chain downstream of it.

Timings are the one metric a component measures about itself rather than about the environment, so
the clock is faked here: what is under test is the bookkeeping around it, never the clock.
"""

from collections.abc import Iterator

import numpy as np
import pytest

from brrrl.metrics.timer import AggregatedElapsed, Timer


class FakeClock:
    """A perf_counter that only moves when a test says so."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeClock]:
    fake = FakeClock()
    monkeypatch.setattr("brrrl.metrics.timer.time.perf_counter", fake)
    yield fake


def test_a_block_dumps_under_the_key_it_was_given(clock: FakeClock) -> None:
    timer = Timer()

    with timer("rollout_time"):
        clock.tick(2.0)

    assert timer.dump() == {"rollout_time": [2.0]}


def test_a_key_timed_repeatedly_keeps_every_sample(clock: FakeClock) -> None:
    """A rollout loop times the same block once an act step, and the window downstream summarises
    the pile. Folding here as well would hand it a mean of means."""
    timer = Timer()

    for seconds in (1.0, 2.0, 3.0):
        with timer("act_time"):
            clock.tick(seconds)

    assert timer.dump() == {"act_time": [1.0, 2.0, 3.0]}


def test_blocks_nest_without_the_inner_one_stealing_the_outer_start(clock: FakeClock) -> None:
    """The interesting timings are a whole rollout and the act inside it, measured together."""
    timer = Timer()

    with timer("rollout_time"):
        clock.tick(1.0)
        with timer("act_time"):
            clock.tick(2.0)
        clock.tick(4.0)

    assert timer.dump() == {"rollout_time": [7.0], "act_time": [2.0]}


def test_a_block_that_raises_is_still_recorded(clock: FakeClock) -> None:
    """A component that dies mid-step logs on the way out, and the timing up to the throw is the
    part worth having."""
    timer = Timer()

    with pytest.raises(RuntimeError), timer("step_time"):
        clock.tick(3.0)
        raise RuntimeError("env died")

    assert timer.dump() == {"step_time": [3.0]}


def test_dumping_clears_so_the_next_window_holds_only_what_it_timed(clock: FakeClock) -> None:
    """Two dumps in a row are two windows. Carrying the first into the second would log the same
    seconds twice and weight them into the mean twice over."""
    timer = Timer()

    with timer("rollout_time"):
        clock.tick(1.0)
    timer.dump()
    with timer("rollout_time"):
        clock.tick(5.0)

    assert timer.dump() == {"rollout_time": [5.0]}


def test_an_empty_window_dumps_nothing_rather_than_keys_with_no_seconds() -> None:
    assert Timer().dump() == {}


def test_an_aggregated_timer_dumps_the_mean_and_holds_no_samples(clock: FakeClock) -> None:
    """The same loop on a long-lived component, where every sample is memory that never comes back.
    The dumped value is the same number the samples would have summarised to."""
    seconds = [1.0, 2.0, 6.0]
    timer = Timer(AggregatedElapsed)

    for tick in seconds:
        with timer("act_time"):
            clock.tick(tick)

    assert timer.dump() == {"act_time": pytest.approx(np.mean(seconds))}
