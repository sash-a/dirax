"""What every backend reduces a record to before storing it.

Metrics come off device as arrays with a value per env or per minibatch, while mlflow and wandb
store one number per key per step. This is the single place that gap is closed, so an unwindowed
chain lands what a windowed one would have landed.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from dirax.metrics.scalar import scalars


def test_a_multi_element_value_is_meaned() -> None:
    assert scalars({"return": np.array([1.0, 2.0, 6.0])}) == {"return": pytest.approx(3.0)}


def test_a_scalar_passes_through_at_its_own_value() -> None:
    assert scalars({"lr": 0.25, "version": np.int32(4)}) == {"lr": 0.25, "version": 4.0}


def test_device_arrays_are_read_without_the_caller_pulling_them_off_first() -> None:
    """Learner metrics arrive as jax arrays, and a backend that took them raw would store a repr."""
    assert scalars({"loss": jnp.array([2.0, 4.0])}) == {"loss": pytest.approx(3.0)}


def test_every_value_comes_out_a_python_float() -> None:
    """A numpy scalar reaches mlflow's protobuf and wandb's json as something they have to guess at."""
    values = scalars({"loss": np.float32(1.0), "steps": np.array([2, 4], dtype=np.int64)})

    assert [type(value) for value in values.values()] == [float, float]


def test_integer_values_are_meaned_without_rounding() -> None:
    assert scalars({"length": np.array([1, 2], dtype=np.int32)}) == {"length": pytest.approx(1.5)}


def test_an_empty_value_is_dropped_rather_than_logged_as_nan() -> None:
    """No episode finished this step is an absence. A nan in the series reads as a broken run."""
    assert scalars({"return": np.array([]), "loss": 1.0}) == {"loss": 1.0}


def test_an_empty_record_reduces_to_nothing() -> None:
    assert scalars({}) == {}
