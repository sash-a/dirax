import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(slots=True)
class MetricAccumulator:
    """Streaming (count, sum, sum_sq, min, max) aggregate for one metric stream."""

    count: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0
    min: float = math.inf
    max: float = -math.inf

    def fold(self, x: ArrayLike) -> None:
        """Accumulate every value in ``x`` (array or scalar) into the running aggregate."""
        x = np.asarray(x)
        if x.size == 0:
            return
        self.count += x.size
        self.sum += float(x.sum(dtype=np.float64))
        self.sum_sq += float(np.square(x, dtype=np.float64).sum())
        self.min = min(self.min, float(x.min()))
        self.max = max(self.max, float(x.max()))

    @property
    def mean(self) -> float:
        if not self.count:
            return math.nan
        return self.sum / self.count

    @property
    def std(self) -> float:
        """Population std (ddof=0), matching ``np.std``: sqrt(E[x^2] - E[x]^2)."""
        if not self.count:
            return math.nan
        return math.sqrt(max(0.0, self.sum_sq / self.count - self.mean**2))

    def summary(self, verbose: bool) -> dict[str, float]:
        """``{mean}``, or ``{mean, std, min, max}`` when verbose. Empty if nothing folded."""
        if not self.count:
            return {}
        if not verbose:
            return {"mean": self.mean}
        return {"mean": self.mean, "std": self.std, "min": self.min, "max": self.max}
