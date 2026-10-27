"""The backend a run gets when there is no tracking server, tested through what it prints.

The printed line is the whole product here, so these read it rather than the logger's state.
"""

import pytest
from etils import epath

from dirax.metrics.stdout import StdoutLogger


def test_one_line_per_record_names_the_step_and_every_metric(capsys: pytest.CaptureFixture[str]) -> None:
    """Keys are sorted so consecutive lines are readable down the column rather than by scanning."""
    StdoutLogger().log({"loss": 2.0, "actor/return": 1.0}, 40)

    assert capsys.readouterr().out == "[40] actor/return=1 loss=2\n"


def test_array_metrics_are_meaned_into_the_line(capsys: pytest.CaptureFixture[str]) -> None:
    """Same reduction every backend applies, so watching a run and reading it back later agree."""
    StdoutLogger().log({"return": [1.0, 3.0]}, 0)

    assert capsys.readouterr().out == "[0] return=2\n"


def test_precision_is_the_loggers(capsys: pytest.CaptureFixture[str]) -> None:
    StdoutLogger(precision=2).log({"loss": 1.23456}, 3)

    assert capsys.readouterr().out == "[3] loss=1.2\n"


def test_a_record_with_nothing_in_it_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    """A bare ``[step]`` line would be indistinguishable from a metric that had disappeared."""
    StdoutLogger().log({"return": []}, 7)

    assert capsys.readouterr().out == ""


def test_an_artifact_line_names_where_it_was_filed_and_where_it_came_from(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without a server to browse, the path is the only way to find it again."""
    StdoutLogger().log_artifact(epath.Path("/run/ckpt"), "actor/checkpoints")

    assert capsys.readouterr().out == "[artifact] actor/checkpoints/ckpt <- /run/ckpt\n"


def test_an_artifact_filed_under_no_directory_is_named_by_itself(capsys: pytest.CaptureFixture[str]) -> None:
    StdoutLogger().log_artifact(epath.Path("/run/ckpt"))

    assert capsys.readouterr().out == "[artifact] ckpt <- /run/ckpt\n"


def test_close_is_idempotent_and_says_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    """Chains share it, so it is reached once per chain and must not narrate each time."""
    logger = StdoutLogger()

    logger.close()
    logger.close()

    assert capsys.readouterr().out == ""
