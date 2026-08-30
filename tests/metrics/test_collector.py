"""The one step clock, and the routing table that decides which chain a record travels down.

ADR-0005: cumulative transitions produced by Actors is the only x-axis in a run, and it lives here
rather than in any backend. A recorder reports how many transitions its metrics account for and
never a step, which is what makes a local recorder and one across a cluster interchangeable.
"""

from conftest import RecordingLogger
from etils import epath

from brrrl.metrics import default_logger
from brrrl.metrics.collector import LocalMetricRecorder, MetricCollector, Origin, WindowedRecorder
from brrrl.metrics.logger import NamespacedProcessor


def test_transitions_accumulate_into_the_step_every_origin_is_logged_against(backend: RecordingLogger) -> None:
    """A learner's loss is stamped with the environment interaction that had happened by then, which
    is what makes overlaying it on a return curve mean anything."""
    collector = MetricCollector({Origin.ACTOR: backend, Origin.LEARNER: backend})

    collector.collect(Origin.ACTOR, {"return": 1.0}, 4)
    collector.collect(Origin.LEARNER, {"loss": 2.0})
    collector.collect(Origin.ACTOR, {"return": 3.0}, 6)
    collector.collect(Origin.LEARNER, {"loss": 4.0})

    assert backend.steps == [4, 4, 10, 10]


def test_an_origin_that_produces_no_transitions_does_not_advance_the_clock(backend: RecordingLogger) -> None:
    """Zero is true of a Trainer or an evaluator rather than a sentinel standing in for "unknown"."""
    collector = MetricCollector({Origin.LEARNER: backend})

    collector.collect(Origin.LEARNER, {"loss": 1.0})
    collector.collect(Origin.LEARNER, {"loss": 2.0})

    assert collector.steps == 0
    assert backend.steps == [0, 0]


def test_each_origin_reaches_its_own_chain_and_no_other() -> None:
    """Whether a source is windowed, and what it is namespaced under, is bound into its chain here.
    A record on the wrong chain would be silently renamed and folded on someone else's schedule."""
    actor, learner = RecordingLogger(), RecordingLogger()
    collector = MetricCollector({Origin.ACTOR: actor, Origin.LEARNER: learner})

    collector.collect(Origin.ACTOR, {"return": 1.0}, 8)
    collector.collect(Origin.LEARNER, {"loss": 2.0})

    assert actor.records == [({"return": 1.0}, 8)]
    assert learner.records == [({"loss": 2.0}, 8)]


def test_an_artifact_travels_its_origins_chain_without_touching_the_clock(backend: RecordingLogger) -> None:
    """It names a place rather than a moment, so there is no step for it to advance or be stamped with."""
    collector = MetricCollector({Origin.LEARNER: backend})
    collector.collect(Origin.LEARNER, {"loss": 1.0})

    collector.collect_artifact(Origin.LEARNER, epath.Path("/run/ckpt"), "checkpoints")

    assert backend.artifacts == [(epath.Path("/run/ckpt"), "checkpoints")]
    assert collector.steps == 0


def test_closing_the_collector_closes_every_chain(backend: RecordingLogger) -> None:
    """Three chains over one backend reach its ``close`` three times, which is the whole reason a
    Logger's ``close`` has to be idempotent: whoever built the chains owns their lifetime."""
    collector = MetricCollector({origin: backend for origin in Origin})

    collector.close()

    assert backend.closes == len(Origin)


def test_a_recorder_carries_the_origin_so_a_component_never_names_itself(backend: RecordingLogger) -> None:
    """An Actor holds a recorder and reports metrics and transitions. Which Origin those are, and
    the step they land against, are decided on this side of the seam."""
    collector = MetricCollector({Origin.ACTOR: backend})
    recorder = LocalMetricRecorder(collector, Origin.ACTOR)

    recorder.record({"return": 1.0}, 16)
    recorder.record_artifact(epath.Path("/run/video.mp4"))

    assert backend.records == [({"return": 1.0}, 16)]
    assert backend.artifacts == [(epath.Path("/run/video.mp4"), None)]
    assert collector.steps == 16


def test_an_origin_is_the_namespace_a_chain_is_built_with(backend: RecordingLogger) -> None:
    """Origin is a StrEnum so the routing key and the namespace cannot drift apart, and a chain is
    wired straight from it without a second table mapping one to the other."""
    collector = MetricCollector({Origin.EVAL: NamespacedProcessor(Origin.EVAL, backend)})

    collector.collect(Origin.EVAL, {"return": 1.0})

    assert backend.records == [({"eval/return": 1.0}, 0)]


def test_the_first_record_lands_immediately(backend: RecordingLogger) -> None:
    """A run that showed nothing until its first window closed would be indistinguishable from a
    run that had hung, and the first window is the longest wait of the run."""
    collector = MetricCollector({Origin.ACTOR: backend})

    WindowedRecorder(LocalMetricRecorder(collector, Origin.ACTOR), every_records=1000).record({"return": 1.0})

    assert backend.records == [({"return/mean": 1.0}, 0)]


def test_records_inside_a_window_are_held_and_emitted_once_it_closes(backend: RecordingLogger) -> None:
    """The point of the whole recorder: one series point per window, whatever the act rate is."""
    collector = MetricCollector({Origin.ACTOR: backend})
    windowed = WindowedRecorder(LocalMetricRecorder(collector, Origin.ACTOR), every_records=3)
    windowed.record({"return": 0.0})

    windowed.record({"return": 2.0})
    windowed.record({"return": 6.0})

    assert len(backend.records) == 1

    windowed.record({"return": 4.0})

    assert backend.records[1:] == [({"return/mean": 4.0}, 0)]


def test_a_window_carries_the_transitions_it_spanned_so_the_clock_stays_exact(backend: RecordingLogger) -> None:
    """Folding before the clock must not lose the count the clock is made of, or an x-axis would
    read differently depending on which side of the seam a run happened to fold on."""
    collector = MetricCollector({Origin.ACTOR: backend})
    windowed = WindowedRecorder(LocalMetricRecorder(collector, Origin.ACTOR), every_records=3)

    for _ in range(7):
        windowed.record({"return": 1.0}, 128)
    windowed.flush()

    assert collector.steps == 7 * 128


def test_a_learner_reporting_no_transitions_still_comes_round(backend: RecordingLogger) -> None:
    """Records rather than transitions are the window's unit for exactly this: a Trainer produces
    none, and a window counting them would hold its metrics for the whole run."""
    collector = MetricCollector({Origin.LEARNER: backend})
    windowed = WindowedRecorder(LocalMetricRecorder(collector, Origin.LEARNER), every_records=2)
    windowed.record({"loss": 4.0})

    windowed.record({"loss": 1.0})
    windowed.record({"loss": 3.0})

    assert backend.records[1:] == [({"loss/mean": 2.0}, 0)]


def test_the_summary_is_keyed_by_metric_and_statistic(backend: RecordingLogger) -> None:
    collector = MetricCollector({Origin.ACTOR: backend})
    windowed = WindowedRecorder(LocalMetricRecorder(collector, Origin.ACTOR), every_records=10, verbose=True)

    windowed.record({"return": [1.0, 3.0]})

    assert backend.records == [({"return/mean": 2.0, "return/std": 1.0, "return/min": 1.0, "return/max": 3.0}, 0)]


def test_a_flush_empties_the_accumulators_so_the_next_window_is_its_own(backend: RecordingLogger) -> None:
    """Otherwise every window is the whole run to date, and a series flattens as a run goes on."""
    collector = MetricCollector({Origin.ACTOR: backend})
    windowed = WindowedRecorder(LocalMetricRecorder(collector, Origin.ACTOR), every_records=2)
    windowed.record({"return": 100.0})

    windowed.record({"loss": 1.0})
    windowed.record({"loss": 3.0})

    assert backend.keys == [["return/mean"], ["loss/mean"]]
    assert backend.records[1] == ({"loss/mean": 2.0}, 0)


def test_closing_flushes_the_partial_window(backend: RecordingLogger) -> None:
    """A run's last window is almost never full, and its metrics are the ones being reported."""
    collector = MetricCollector({Origin.ACTOR: backend})
    windowed = WindowedRecorder(LocalMetricRecorder(collector, Origin.ACTOR), every_records=10)
    windowed.record({"return": 0.0})
    windowed.record({"return": 8.0}, 6)

    windowed.close()

    assert backend.records[1:] == [({"return/mean": 8.0}, 6)]
    assert backend.closes == 1


def test_closing_with_nothing_pending_logs_nothing(backend: RecordingLogger) -> None:
    """An empty record at the end of a run would read as a metric that had collapsed to zero."""
    collector = MetricCollector({Origin.ACTOR: backend})
    windowed = WindowedRecorder(LocalMetricRecorder(collector, Origin.ACTOR), every_records=10)
    windowed.record({"return": 1.0})

    windowed.close()

    assert len(backend.records) == 1
    assert backend.closes == 1


def test_closing_twice_flushes_once_and_reaches_the_backend_twice(backend: RecordingLogger) -> None:
    """Chains share a backend, so ``close`` is reached once per chain; it is the backend's own
    ``close`` that has to be idempotent, and a window must not invent a second flush."""
    collector = MetricCollector({Origin.ACTOR: backend})
    windowed = WindowedRecorder(LocalMetricRecorder(collector, Origin.ACTOR), every_records=10)
    windowed.record({"return": 1.0})
    windowed.record({"return": 5.0}, 3)

    windowed.close()
    windowed.close()

    assert backend.steps == [0, 3]
    assert backend.closes == 2


def test_artifacts_skip_the_window(backend: RecordingLogger) -> None:
    """An Artifact names a place rather than a moment, so there is nothing to fold and nothing to
    wait for: a checkpoint held back until the window closed could outlive the process."""
    collector = MetricCollector({Origin.ACTOR: backend})
    windowed = WindowedRecorder(LocalMetricRecorder(collector, Origin.ACTOR), every_records=1000)

    windowed.record_artifact(epath.Path("/run/ckpt"), "checkpoints")

    assert backend.artifacts == [(epath.Path("/run/ckpt"), "checkpoints")]


def test_a_windowed_chain_names_the_origin_the_metric_and_the_statistic(backend: RecordingLogger) -> None:
    """The window folds before the namespace rather than after it, and the series name someone reads
    for the life of the project is the same either way round."""
    actor, _ = default_logger(backend, every_records=10)

    actor.record({"return": [1.0, 3.0]})

    assert backend.records == [({"actor/return/mean": 2.0}, 0)]
