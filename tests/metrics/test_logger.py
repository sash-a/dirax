"""The Processors a metric chain is built out of.

A Processor implements the same protocol as the backend at the end of the chain, so each of these
is tested by reading what reached a RecordingLogger standing in for that backend. Windowing is
where the interesting arithmetic is: ADR-0005 puts it below the counting sink precisely so a
boundary is arithmetic on a step that is already resolved, and these pin that arithmetic down.
"""

from collections.abc import Callable

import pytest
from conftest import RecordingLogger
from etils import epath

from brrrl.metrics.logger import Logger, NamespacedProcessor, WindowedProcessor


def test_namespacing_prefixes_every_key_and_leaves_the_step_alone(backend: RecordingLogger) -> None:
    """The namespace is the Origin's, fixed at wiring time, so a component never names itself."""
    NamespacedProcessor("actor", backend).log({"return": 1.0, "length": 2.0}, 40)

    assert backend.records == [({"actor/return": 1.0, "actor/length": 2.0}, 40)]


def test_an_artifact_is_filed_under_the_namespace_by_default(backend: RecordingLogger) -> None:
    NamespacedProcessor("learner", backend).log_artifact(epath.Path("/run/ckpt"))

    assert backend.artifacts == [(epath.Path("/run/ckpt"), "learner")]


def test_an_artifacts_own_directory_nests_inside_the_namespace(backend: RecordingLogger) -> None:
    """Same shape as the keys: the Origin's namespace is the root of everything it files."""
    NamespacedProcessor("learner", backend).log_artifact(epath.Path("/run/ckpt"), "checkpoints")

    assert backend.artifacts == [(epath.Path("/run/ckpt"), "learner/checkpoints")]


def test_namespacing_forwards_the_close(backend: RecordingLogger) -> None:
    NamespacedProcessor("actor", backend).close()

    assert backend.closes == 1


def test_the_first_record_lands_immediately(backend: RecordingLogger) -> None:
    """A run that showed nothing until its first window closed would be indistinguishable from a
    run that had hung, and the first window is the longest wait of the run."""
    WindowedProcessor(backend, every=1000).log({"return": 1.0}, 0)

    assert backend.records == [({"return/mean": 1.0}, 0)]


def test_records_inside_a_window_are_held_and_emitted_once_it_closes(backend: RecordingLogger) -> None:
    """The point of the whole processor: one series point per window, whatever the act rate is."""
    windowed = WindowedProcessor(backend, every=10)
    windowed.log({"return": 0.0}, 0)

    windowed.log({"return": 2.0}, 4)
    windowed.log({"return": 6.0}, 8)

    assert backend.steps == [0]

    windowed.log({"return": 4.0}, 10)

    assert backend.records[1:] == [({"return/mean": 4.0}, 10)]


def test_the_summary_is_keyed_by_metric_and_statistic(backend: RecordingLogger) -> None:
    windowed = WindowedProcessor(backend, every=10, verbose=True)

    windowed.log({"return": [1.0, 3.0]}, 0)

    assert backend.records == [({"return/mean": 2.0, "return/std": 1.0, "return/min": 1.0, "return/max": 3.0}, 0)]


def test_a_flush_empties_the_accumulators_so_the_next_window_is_its_own(backend: RecordingLogger) -> None:
    """Otherwise every window is the whole run to date, and a series flattens as a run goes on."""
    windowed = WindowedProcessor(backend, every=10)
    windowed.log({"return": 100.0}, 0)

    windowed.log({"loss": 1.0}, 5)
    windowed.log({"loss": 3.0}, 10)

    assert backend.keys == [["return/mean"], ["loss/mean"]]
    assert backend.records[1] == ({"loss/mean": 2.0}, 10)


def test_a_window_that_closes_late_rebases_onto_its_own_boundary(backend: RecordingLogger) -> None:
    """Steps are transitions produced, so they arrive in strides of a whole rollout and land past
    the boundary rather than on it. Counting ``every`` from where the flush happened would drift
    the boundaries by the overshoot, so each flush snaps back onto the grid."""
    windowed = WindowedProcessor(backend, every=10)
    windowed.log({"return": 0.0}, 0)

    windowed.log({"return": 1.0}, 47)
    windowed.log({"return": 2.0}, 48)
    windowed.log({"return": 3.0}, 50)

    assert backend.steps == [0, 47, 50]


def test_closing_flushes_the_partial_window_at_the_last_step_it_saw(backend: RecordingLogger) -> None:
    """A run's last window is almost never full, and its metrics are the ones being reported."""
    windowed = WindowedProcessor(backend, every=10)
    windowed.log({"return": 0.0}, 0)
    windowed.log({"return": 8.0}, 6)

    windowed.close()

    assert backend.records[1:] == [({"return/mean": 8.0}, 6)]
    assert backend.closes == 1


def test_closing_with_nothing_pending_logs_nothing(backend: RecordingLogger) -> None:
    """An empty record at the end of a run would read as a metric that had collapsed to zero."""
    windowed = WindowedProcessor(backend, every=10)
    windowed.log({"return": 1.0}, 0)

    windowed.close()

    assert len(backend.records) == 1
    assert backend.closes == 1


def test_closing_twice_flushes_once_and_reaches_the_backend_twice(backend: RecordingLogger) -> None:
    """Chains share a backend, so ``close`` is reached once per chain; it is the backend's own
    ``close`` that has to be idempotent, and a Processor must not invent a second flush."""
    windowed = WindowedProcessor(backend, every=10)
    windowed.log({"return": 1.0}, 0)
    windowed.log({"return": 5.0}, 3)

    windowed.close()
    windowed.close()

    assert backend.steps == [0, 3]
    assert backend.closes == 2


def test_artifacts_skip_the_window(backend: RecordingLogger) -> None:
    """An Artifact names a place rather than a moment, so there is nothing to fold and nothing to
    wait for: a checkpoint held back until the window closed could outlive the process."""
    windowed = WindowedProcessor(backend, every=1000)

    windowed.log_artifact(epath.Path("/run/ckpt"), "checkpoints")

    assert backend.artifacts == [(epath.Path("/run/ckpt"), "checkpoints")]


@pytest.mark.parametrize(
    "compose",
    [
        pytest.param(lambda backend: NamespacedProcessor("actor", WindowedProcessor(backend, every=10)), id="outside"),
        pytest.param(lambda backend: WindowedProcessor(NamespacedProcessor("actor", backend), every=10), id="inside"),
    ],
)
def test_a_namespaced_window_names_the_origin_the_metric_and_the_statistic(
    backend: RecordingLogger, compose: Callable[[RecordingLogger], Logger]
) -> None:
    """Both compositions have to agree, because which one a chain is built as is a wiring detail
    and the key it produces is a series name someone reads for the life of the project."""
    chain = compose(backend)

    chain.log({"return": [1.0, 3.0]}, 0)

    assert backend.records == [({"actor/return/mean": 2.0}, 0)]
