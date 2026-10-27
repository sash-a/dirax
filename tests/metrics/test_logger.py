"""The Processors a metric chain is built out of.

A Processor implements the same protocol as the backend at the end of the chain, so each of these
is tested by reading what reached a RecordingLogger standing in for that backend. Windowing is not
one of them: it folds before the clock rather than after it, so it is a recorder and is tested
beside the collector it sits in front of.
"""

from conftest import RecordingLogger
from etils import epath

from dirax.metrics.logger import NamespacedProcessor


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
