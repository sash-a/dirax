import numpy as np

from dirax.types import Metrics


def scalars(metrics: Metrics) -> dict[str, float]:
    """One float per metric, as backends store them.

    Multi-element values are meaned — an unwindowed chain lands what a windowed one
    would have. Empty ones are dropped rather than logged as nan.
    """
    arrays = ((key, np.asarray(value)) for key, value in metrics.items())
    return {key: float(array.mean(dtype=np.float64)) for key, array in arrays if array.size}
