"""Nested metrics into the flat keys a backend stores.

Algorithms build their metrics as trees because that is how the code that computes them is
grouped. Backends take a flat namespace, so the join happens once, here, and never in an algorithm.
"""

import numpy as np
import pytest

from brrrl.metrics.flatten import flatten


def test_nested_keys_are_joined_into_one_path() -> None:
    assert flatten({"loss": {"policy": 1.0, "value": 2.0}, "lr": 3.0}) == {
        "loss/policy": 1.0,
        "loss/value": 2.0,
        "lr": 3.0,
    }


def test_the_join_reaches_all_the_way_down() -> None:
    assert flatten({"a": {"b": {"c": 1.0}}}) == {"a/b/c": 1.0}


def test_an_already_flat_mapping_is_unchanged() -> None:
    assert flatten({"lr": 0.1, "steps": 3}) == {"lr": 0.1, "steps": 3}


def test_the_separator_is_the_callers() -> None:
    assert flatten({"a": {"b": 1.0}}, separator=".") == {"a.b": 1.0}


def test_leaves_are_forwarded_as_they_are() -> None:
    """Arrays go through untouched: reducing them is the backend chain's job, and doing it here
    would mean an unwindowed chain had already thrown the spread away."""
    values = np.array([1.0, 2.0])

    assert flatten({"episode": {"return": values}})["episode/return"] is values


def test_empty_branches_vanish() -> None:
    """A branch a component computed nothing into is not a metric, and would land as an empty key."""
    assert flatten({"eval": {}, "loss": 1.0}) == {"loss": 1.0}
    assert flatten({"eval": {"episode": {}}}) == {}


@pytest.mark.parametrize(
    "colliding",
    [
        pytest.param({"a": {"b": 1.0}, "a/b": 2.0}, id="nested-then-flat"),
        pytest.param({"a/b": 2.0, "a": {"b": 1.0}}, id="flat-then-nested"),
    ],
)
def test_two_branches_writing_one_key_is_refused(colliding: dict[str, object]) -> None:
    """Silently keeping one of them would drop a metric a run is being judged on, and the
    disagreement is in the caller's tree rather than anything a backend could report."""
    with pytest.raises(ValueError, match="a/b"):
        flatten(colliding)
