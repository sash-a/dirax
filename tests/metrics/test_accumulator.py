"""The arithmetic a window's summary is made of.

An accumulator sees every value a source produced between two flushes, so it holds an aggregate
rather than the values. What it must never do is disagree with the numpy the same values would
have gone through in one array.
"""

import math

import numpy as np
import pytest

from brrrl.metrics.accumulator import MetricAccumulator


def test_folding_arrays_and_scalars_agrees_with_numpy_over_everything_folded() -> None:
    """Records arrive one per act step and each carries a value per env, so a window is a ragged
    pile of shapes. The aggregate has to be what concatenating them would have given."""
    folds = [np.array([1.0, 2.0, 3.0]), 4.0, np.array([[5.0, 6.0], [7.0, 8.0]])]
    accumulator = MetricAccumulator()

    for fold in folds:
        accumulator.fold(fold)

    everything = np.concatenate([np.asarray(fold).ravel() for fold in folds])
    assert accumulator.count == everything.size
    assert accumulator.mean == pytest.approx(everything.mean())
    assert accumulator.std == pytest.approx(everything.std())
    assert accumulator.min == everything.min()
    assert accumulator.max == everything.max()


def test_the_std_is_the_population_std_rather_than_a_sample_one() -> None:
    """A window holds every value in it, not a sample of them, so ddof=0 is the honest one."""
    values = np.array([1.0, 2.0, 3.0, 4.0])
    accumulator = MetricAccumulator()

    accumulator.fold(values)

    assert accumulator.std == pytest.approx(np.std(values, ddof=0))
    assert accumulator.std != pytest.approx(np.std(values, ddof=1))


def test_a_constant_stream_far_from_zero_has_no_negative_variance() -> None:
    """E[x^2] - E[x]^2 cancels catastrophically on large equal values, and a nan std would reach a
    backend as a broken series rather than as the zero variance it is."""
    accumulator = MetricAccumulator()

    for _ in range(4):
        accumulator.fold(np.float32(1e8))

    assert accumulator.std == 0.0


def test_an_empty_array_folds_to_nothing() -> None:
    """An env that finished no episodes this step reports an empty array, which is not a zero."""
    accumulator = MetricAccumulator()
    accumulator.fold([2.0, 4.0])

    accumulator.fold(np.array([]))

    assert accumulator.count == 2
    assert accumulator.mean == pytest.approx(3.0)
    assert accumulator.min == 2.0


def test_min_and_max_span_every_fold_rather_than_the_last() -> None:
    accumulator = MetricAccumulator()

    accumulator.fold([-3.0, 0.0])
    accumulator.fold([7.0])
    accumulator.fold([1.0])

    assert (accumulator.min, accumulator.max) == (-3.0, 7.0)


def test_integer_values_do_not_round_the_mean() -> None:
    """Counts are metrics too, and their window mean is rarely an integer."""
    accumulator = MetricAccumulator()

    accumulator.fold(np.array([1, 2], dtype=np.int32))

    assert accumulator.mean == pytest.approx(1.5)


def test_nothing_folded_summarises_as_nothing() -> None:
    """A key nobody reported this window must vanish from the flush, not land as a nan."""
    accumulator = MetricAccumulator()

    assert accumulator.summary(verbose=False) == {}
    assert accumulator.summary(verbose=True) == {}
    assert math.isnan(accumulator.mean)
    assert math.isnan(accumulator.std)


def test_a_summary_is_the_mean_alone_unless_it_is_verbose() -> None:
    """Four series per metric is a lot to pay for every metric in a run, so spread is opt-in."""
    accumulator = MetricAccumulator()
    accumulator.fold([1.0, 3.0])

    assert accumulator.summary(verbose=False) == {"mean": 2.0}
    assert accumulator.summary(verbose=True) == {"mean": 2.0, "std": 1.0, "min": 1.0, "max": 3.0}
