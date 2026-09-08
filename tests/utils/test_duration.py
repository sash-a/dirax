"""What a Duration promises the loops that gate on it.

A checkpointer and a metric window ask the same question once a tick, so most of what matters here
is shared rather than per-implementation: a Duration answers rather than acts, which is what lets
one value serve several callers and lets a caller ask twice without spending anything. `Instant` is
built by hand throughout — the pair of clocks is data, so nothing here needs a fake one.
"""

import pytest

from brrrl.utils.duration import Duration, EverySeconds, EveryTicks, Instant

DURATIONS: tuple[type[Duration], ...] = (EveryTicks, EverySeconds)
"""Every Duration in the module. Typed, so `ty` rejects one that does not answer ``due``."""

A_LONG_TIME = 3600.0
"""Enough seconds that a tick-based case is never the thing under test, and vice versa."""


def ticks(step: int) -> Instant:
    """An Instant that has moved only on the count."""
    return Instant(step, 0.0)


def seconds(time: float) -> Instant:
    """An Instant that has moved only on the clock."""
    return Instant(0, time)


@pytest.fixture(params=["ticks", "seconds"])
def duration_and_span(request: pytest.FixtureRequest) -> tuple[Duration, Instant, Instant]:
    """Each Duration with a ``prev`` and a ``now`` exactly one span apart on its own clock."""
    if request.param == "ticks":
        return EveryTicks(10), ticks(0), ticks(10)
    return EverySeconds(10.0), seconds(0.0), seconds(10.5)


def test_a_whole_span_is_due(duration_and_span: tuple[Duration, Instant, Instant]) -> None:
    every, prev, now = duration_and_span

    assert every.due(now, prev) is True


def test_asking_twice_gives_the_same_answer(duration_and_span: tuple[Duration, Instant, Instant]) -> None:
    """The property the whole design rests on. Orbax asks its save policy twice for one step, and a
    Duration that spent itself on the first ask would answer no to the second and never save."""
    every, prev, now = duration_and_span

    assert [every.due(now, prev) for _ in range(3)] == [True, True, True]


def test_one_duration_serves_several_callers(duration_and_span: tuple[Duration, Instant, Instant]) -> None:
    """An actor's window and a learner's are built from one ``every``. Holding a position rather
    than answering a question would have the first caller to come round spend the second's turn."""
    every, prev, now = duration_and_span
    other_prev, other_now = prev, prev  # A second caller that has not moved at all.

    assert every.due(now, prev) is True
    assert every.due(other_now, other_prev) is False
    assert every.due(now, prev) is True


def test_ticks_are_due_on_the_span_and_not_inside_it() -> None:
    """``n`` ticks means ``n``, so the tick that completes the span is the one that acts."""
    every_ten = EveryTicks(10)

    assert [step for step in range(1, 12) if every_ten.due(ticks(step), ticks(0))] == [10, 11]


def test_a_span_of_one_tick_is_due_every_tick() -> None:
    every_tick = EveryTicks(1)

    assert all(every_tick.due(ticks(step + 1), ticks(step)) for step in range(4))


def test_ticks_read_the_count_and_never_the_clock() -> None:
    """The point of the pair: a save gated on updates must not come round because a run was slow."""
    every_ten = EveryTicks(10)

    assert every_ten.due(Instant(1, A_LONG_TIME), Instant(0, 0.0)) is False


def test_a_gap_that_skips_a_span_is_due_once_rather_than_once_per_span() -> None:
    """A loop reports the tick it is on, not every tick since. Since the caller moves ``prev`` to
    where it acted, the spans it jumped are gone rather than owed."""
    every_ten = EveryTicks(10)
    acted_at = ticks(100)  # Came round 100 ticks late, so ten spans went by.

    assert every_ten.due(ticks(100), ticks(0)) is True
    assert [step for step in range(101, 111) if every_ten.due(ticks(step), acted_at)] == [110]


@pytest.mark.parametrize("n", [0, -1])
def test_a_span_shorter_than_one_tick_is_rejected(n: int) -> None:
    """Zero would divide a loop into spans it is never inside, and answer due to every ask."""
    with pytest.raises(ValueError, match="at least one tick"):
        EveryTicks(n)


def test_seconds_are_due_once_the_span_has_passed_and_not_before() -> None:
    every_ten = EverySeconds(10.0)

    assert every_ten.due(seconds(9.9), seconds(0.0)) is False
    assert every_ten.due(seconds(10.1), seconds(0.0)) is True


def test_seconds_read_the_clock_and_never_the_count() -> None:
    """A sink publishing on a timer must not speed up because the learner did."""
    every_ten = EverySeconds(10.0)

    assert every_ten.due(Instant(1_000_000, 0.0), Instant(0, 0.0)) is False


def test_an_instant_reads_both_clocks_at_once() -> None:
    """``now`` is what a caller passes to whichever Duration it was given, so it carries both."""
    before = Instant.now(step=7)

    assert before.step == 7
    assert Instant.now(step=8).time >= before.time
