"""The one step clock, and the routing table that decides which chain a record travels down.

ADR-0005: cumulative transitions produced by Actors is the only x-axis in a run, and it lives here
rather than in any backend. A recorder reports how many transitions its metrics account for and
never a step, which is what makes a local recorder and one across a cluster interchangeable.
"""

from conftest import RecordingLogger
from etils import epath

from brrrl.metrics.collector import LocalMetricRecorder, MetricCollector, Origin
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
