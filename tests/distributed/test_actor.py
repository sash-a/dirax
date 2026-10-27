"""An Actor's whole job: run a Roller in a loop, ship what it produced, say what it cost, let go.

There is no algorithm in here to get wrong, so what is worth holding an Actor to is its lifecycle.
One that leaks a Roller leaves a pool of environment subprocesses behind on every run; one that
frees its replay client from the wrong thread frees it out from under a send still in flight; and
one that reports the wrong count of transitions moves every curve in the run along an x-axis that
has stopped agreeing with the interaction underneath it.

The Roller and the replay client are doubles: what they do is settled in their own tests, and the
loop around them is the subject here. A Roller is also the only thing that can bound the loop —
`Actor.run` runs until it is told to stop — so the rollout that asks for the stop is the same
rollout a test then checks was finished rather than abandoned.

`Actor.RAY_OPTIONS` reserves a second thread for exactly one caller: something asking a running
Actor to stop. The last test here is that caller, and it is the shape a real run's ``close`` has.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, cast

import pytest
from conftest import Recorded, settled

from dirax.distributed.components.actor import Actor
from dirax.types import Metrics

TRANSITIONS = 12
"""What one rollout of the double costs. Neither a step count nor an env count, so a confusion
between the three is a different number here."""


class Bail(Exception):
    """What a Roller raises when a test wants to see an Actor come apart mid-run."""


@dataclass
class Rolling:
    """A Roller that hands back a numbered rollout and remembers whether it was released.

    ``stop`` is wired to the Actor's own ``close`` by `build`, and called on the rollout named by
    ``stop_after``: the loop has no other end, and asking for the stop from inside a rollout is
    what makes "after the rollout it is on" an observable claim rather than a timing accident.
    """

    num_transitions: int = TRANSITIONS
    stop_after: int | None = None
    raise_after: int | None = None
    stop: Callable[[], None] = lambda: None
    rollouts: int = 0
    closed: bool = False

    def rollout(self) -> tuple[int, Metrics]:
        self.rollouts += 1
        if self.rollouts == self.raise_after:
            raise Bail(f"rollout {self.rollouts}")
        if self.rollouts == self.stop_after:
            self.stop()
        return self.rollouts, {"episode_return": float(self.rollouts)}

    def close(self) -> None:
        self.closed = True


@dataclass
class Sending:
    """A replay client that keeps what it was asked to send, and says when it was released."""

    sent: list[int] = field(default_factory=list)
    closed: bool = False

    def send(self, rollout: int) -> None:
        self.sent.append(rollout)

    def close(self) -> None:
        self.closed = True


class InProcess(Protocol):
    """An Actor as a test holds one, which is not the view `ray.method` gives its methods.

    That decorator leaves the function alone and declares the attribute to be the handle a placed
    Component is reached through, which is what makes ``actor.run.remote()`` check against an
    `ActorProxy` and what makes calling one here look like calling a handle. Naming the surface a
    test uses says so once instead of at every call.
    """

    stop: bool

    def run(self) -> None: ...
    def close(self) -> None: ...


@dataclass
class Wired:
    """One Actor and the three collaborators it was built over, which is where the assertions are."""

    actor: InProcess
    roller: Rolling
    client: Sending
    recorder: Recorded


def build(roller: Rolling | None = None, recorder: Recorded | None = None) -> Wired:
    """Rollers and clients are built inside the Actor rather than shipped to it, so a test hands it
    factories over doubles it already holds — which is also how a real run hands it a Roller it
    could not have pickled."""
    roller = Rolling() if roller is None else roller
    client, recorder = Sending(), Recorded() if recorder is None else recorder
    actor = cast(InProcess, Actor(lambda: roller, lambda: client, recorder))  # ty: ignore[invalid-argument-type]
    roller.stop = actor.close
    return Wired(actor, roller, client, recorder)


def test_every_rollout_an_actor_takes_is_shipped_and_reported() -> None:
    """Nothing between the Roller and the replay client is allowed to drop a Trajectory: a rollout
    that was taken and not sent is environment interaction the learner paid for and never saw."""
    wired = build(Rolling(stop_after=3))

    wired.actor.run()

    assert wired.roller.rollouts == 3
    assert wired.client.sent == [1, 2, 3]
    assert [metrics["episode_return"] for metrics, _ in wired.recorder.records] == [1.0, 2.0, 3.0]


def test_an_actor_accounts_its_metrics_to_the_transitions_its_roller_says_it_took() -> None:
    """The count is the Roller's own, and it is the only thing that advances the run's step clock.
    An Actor that passed its rollout count, or its env count, would leave every metric in the run
    plotted against a number that is not environment interaction (ADR-0005)."""
    wired = build(Rolling(stop_after=2))

    wired.actor.run()

    assert wired.recorder.transitions == [TRANSITIONS, TRANSITIONS]


def test_close_stops_the_loop_after_the_rollout_it_is_on_rather_than_during_it() -> None:
    """The third rollout asks for the stop halfway through itself. It still has to be sent and
    reported — a Trajectory abandoned between the Roller and the wire is interaction thrown away —
    and no fourth may be taken."""
    wired = build(Rolling(stop_after=3))

    wired.actor.run()

    assert wired.roller.rollouts == 3
    assert wired.client.sent[-1] == 3
    assert len(wired.recorder.records) == 3


def test_close_asks_the_loop_to_stop_and_releases_nothing_itself() -> None:
    """Its caller is on the Actor's other thread, so anything it freed it would free out from under
    a send still in flight. Letting go is ``run``'s own job, on its way out."""
    wired = build()

    wired.actor.close()

    assert wired.actor.stop
    assert not wired.roller.closed
    assert not wired.client.closed
    assert wired.recorder.closes == 0


def test_an_actor_that_has_left_its_loop_has_released_everything_it_held() -> None:
    """A Roller holds environment subprocesses and a replay client holds a socket, and both outlive
    the Actor if nothing releases them."""
    wired = build(Rolling(stop_after=1))

    wired.actor.run()

    assert wired.roller.closed
    assert wired.client.closed
    assert wired.recorder.closes == 1


def test_a_rollout_that_raises_releases_everything_the_actor_held_and_is_not_swallowed() -> None:
    """A run comes apart for reasons an Actor has no say in. The driver has to hear about it — a
    dead Actor that reported nothing is a run that quietly slows down — and the subprocesses behind
    the Roller have to go with it."""
    wired = build(Rolling(raise_after=2))

    with pytest.raises(Bail):
        wired.actor.run()

    assert wired.roller.rollouts == 2
    assert wired.roller.closed
    assert wired.client.closed
    assert wired.recorder.closes == 1


def test_an_actor_asked_to_stop_from_another_thread_leaves_its_loop() -> None:
    """The shape a real teardown has: ``run`` holds a thread for the life of the run and ``close``
    arrives on the second one Ray reserved for it. Nothing here is scripted from inside the loop,
    so this is also the one test that the flag is read at all rather than merely set."""
    wired = build()
    running = threading.Thread(target=wired.actor.run, name="actor-run")
    running.start()

    settled(lambda: wired.roller.rollouts > 0)
    wired.actor.close()
    running.join(timeout=5.0)

    assert not running.is_alive()
    assert wired.client.sent == list(range(1, wired.roller.rollouts + 1))
    assert wired.roller.closed
