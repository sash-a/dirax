from collections.abc import Mapping

from numpy.typing import ArrayLike

from dirax.types import Metrics, NestedMetrics


def flatten(metrics: NestedMetrics, separator: str = "/") -> Metrics:
    """Join nested keys: ``{"a": {"b": x}}`` becomes ``{"a/b": x}``.

    Empty branches vanish. Collisions raise rather than silently dropping a metric.
    """
    flat: dict[str, ArrayLike] = {}
    for key, value in metrics.items():
        branch = (
            {f"{key}{separator}{nested}": leaf for nested, leaf in flatten(value, separator).items()}
            if isinstance(value, Mapping)
            else {key: value}
        )
        collisions = flat.keys() & branch.keys()
        if collisions:
            raise ValueError(f"flatten() would write {sorted(collisions)} twice")
        flat |= branch
    return flat
