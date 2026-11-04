"""What an ActorMonitor owes a run whose actors come apart: restarts that are bounded, and errors
that are neither lost nor mistaken for a reason to restart.

The actors are real Ray actors, because what the monitor reacts to is what Ray hands back when
one fails — a `RayTaskError` for a ``run`` that raised, a `RayActorError` for a process that
died — and a double would only restate what the monitor assumes. The monitor itself runs in this
process, over a `Recorded` it can be asked about, and on a thread so that a monitor which never
returns fails a test rather than hanging the suite.
"""

import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import ray
from conftest import Recorded, settled
from flaky import Flaky
from ray.exceptions import RayActorError, RayTaskError

from dirax.distributed.components.actor import ActorMonitor

JOIN = 30.0
"""Generous enough for a restarted worker to import its module cold."""

ALWAYS = 10**6
"""More failed runs than any test lets an actor have."""


@pytest.fixture(scope="module")
def cluster() -> Iterator[None]:
    job = {"PYTHONPATH": str(Path(__file__).parent)}
    ray.init(num_cpus=4, num_gpus=0, include_dashboard=False, log_to_driver=False, runtime_env={"env_vars": job})
    yield
    ray.shutdown()


@dataclass
class Monitoring:
    """A monitor running on its own thread, and whatever it ended with."""

    monitor: ActorMonitor
    thread: threading.Thread
    raised: list[ExceptionGroup] = field(default_factory=list)

    def finish(self) -> ExceptionGroup | None:
        self.thread.join(timeout=JOIN)
        assert not self.thread.is_alive(), "the monitor never returned"
        return self.raised[0] if self.raised else None


def start(actors: list, max_restarts: int, recorder: Recorded) -> Monitoring:
    monitor = ActorMonitor(actors, max_restarts, recorder)
    raised: list[ExceptionGroup] = []

    def run() -> None:
        try:
            monitor.run()
        except ExceptionGroup as e:
            raised.append(e)

    thread = threading.Thread(target=run, name="monitor", daemon=True)
    thread.start()
    return Monitoring(monitor, thread, raised)


def place(tmp_path: Path, name: str, **behaviour: object):
    ledger = tmp_path / name
    ledger.touch()
    actor = ray.remote(num_cpus=0, max_concurrency=2, max_restarts=-1)(Flaky).remote(str(ledger), **behaviour)
    return actor, ledger


def starts(ledger: Path) -> int:
    return len(ledger.read_text().splitlines())


@pytest.fixture(params=[False, True], ids=["raises", "crashes"])
def crash(request: pytest.FixtureRequest) -> bool:
    """A run that raises leaves its process up and one that crashes takes it down, which reach the
    monitor as different exceptions and have to be brought back differently."""
    return request.param


def test_an_actor_that_always_fails_is_restarted_up_to_its_budget_and_then_the_run_is_brought_down(
    cluster: None, tmp_path: Path, crash: bool
) -> None:
    """The budget has to end somewhere, or a broken environment restarts forever. When it does,
    the healthy actor beside it is closed — the monitor returns only once every ``run`` has — and
    the failure that ended it is reported. The ones it restarted away are only logged."""
    max_restarts = 2
    broken, broken_ledger = place(tmp_path, "broken", fail_first=ALWAYS, crash=crash)
    healthy, healthy_ledger = place(tmp_path, "healthy")
    recorder = Recorded()

    failure = start([broken, healthy], max_restarts, recorder).finish()

    assert isinstance(failure, ExceptionGroup)
    expected = RayActorError if crash else RayTaskError
    assert len(failure.exceptions) == 1
    assert all(isinstance(e, expected) for e in failure.exceptions)
    assert starts(broken_ledger) == max_restarts + 1
    assert starts(healthy_ledger) == 1
    assert [metrics for metrics, _ in recorder.records] == [{"actor_restarts": n} for n in (1, 2)]


def test_the_budget_is_each_actors_own(cluster: None, tmp_path: Path) -> None:
    """A fleet of actors that each fail once is not a fleet that has failed ``len`` times."""
    first, first_ledger = place(tmp_path, "first", fail_first=2)
    second, second_ledger = place(tmp_path, "second", fail_first=2)
    monitoring = start([first, second], 2, Recorded())

    settled(lambda: starts(first_ledger) == 3 and starts(second_ledger) == 3, timeout=JOIN)
    monitoring.monitor.close()

    assert monitoring.finish() is None
    assert monitoring.monitor.num_restarts == 4


def test_an_actor_that_fails_once_is_restarted_and_keeps_running(cluster: None, tmp_path: Path, crash: bool) -> None:
    """The point of the monitor: one bad rollout, or one OOM-killed worker, costs a restart and
    not the run. The actor beside it is never disturbed."""
    flaky, flaky_ledger = place(tmp_path, "flaky", fail_first=1, crash=crash)
    steady, steady_ledger = place(tmp_path, "steady")
    recorder = Recorded()
    monitoring = start([flaky, steady], 3, recorder)

    settled(lambda: starts(flaky_ledger) == 2)
    still_running = monitoring.thread.is_alive()
    monitoring.monitor.close()
    monitoring.finish()

    assert still_running
    assert starts(flaky_ledger) == 2
    assert starts(steady_ledger) == 1
    assert [metrics for metrics, _ in recorder.records] == [{"actor_restarts": 1}]


def test_an_actor_that_failed_once_and_recovered_does_not_fail_the_run(
    cluster: None, tmp_path: Path, crash: bool
) -> None:
    """A failure that was restarted away is something the run got past, so a clean shutdown after
    it is a clean run rather than an ``ExceptionGroup`` for the one that is long forgotten."""
    flaky, flaky_ledger = place(tmp_path, "flaky", fail_first=1, crash=crash)
    monitoring = start([flaky], 3, Recorded())

    settled(lambda: starts(flaky_ledger) == 2)
    monitoring.monitor.close()

    assert monitoring.finish() is None


def test_an_actor_that_fails_while_the_run_is_stopping_is_reported_and_not_restarted(
    cluster: None, tmp_path: Path
) -> None:
    """Once the run has asked for the stop, a restarted actor would only have to be stopped again.
    What it died of is still worth hearing, so it comes back with the shutdown."""
    leaving, leaving_ledger = place(tmp_path, "leaving", fail_on_stop=True)
    other, other_ledger = place(tmp_path, "other", fail_on_stop=True)
    recorder = Recorded()
    monitoring = start([leaving, other], 3, recorder)

    settled(lambda: starts(leaving_ledger) == 1 and starts(other_ledger) == 1)
    monitoring.monitor.close()
    failure = monitoring.finish()

    assert isinstance(failure, ExceptionGroup)
    assert len(failure.exceptions) == 2
    assert all(isinstance(e, RayTaskError) for e in failure.exceptions)
    assert starts(leaving_ledger) == 1
    assert starts(other_ledger) == 1
    assert recorder.records == []


def test_a_clean_shutdown_returns_without_error(cluster: None, tmp_path: Path) -> None:
    actors = [place(tmp_path, f"actor{i}")[0] for i in range(2)]
    monitoring = start(actors, 3, Recorded())

    settled(lambda: monitoring.thread.is_alive())
    monitoring.monitor.close()

    assert monitoring.finish() is None
