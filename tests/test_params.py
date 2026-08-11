"""The write side of the parameter path, which is the run's one monotonic authority for a Version.

The read side is asserted through the Policy that sits at the end of the chain, in
``tests/acting/test_policy.py``.
"""

import jax.numpy as jnp

from brrrl.params import InMemoryParamStore
from brrrl.types import Params, VersionedParams


def params(scale: float) -> Params:
    return {"params": {"scale": jnp.float32(scale)}}


def seeded(version: int = 0) -> InMemoryParamStore:
    return InMemoryParamStore(VersionedParams(params(1.0), version))


def test_the_write_side_assigns_the_version() -> None:
    """A publisher that passed its own count could repeat one, and staleness would stop meaning anything."""
    store = seeded()

    assert store.publish(params(2.0)) == 1
    assert store.publish(params(3.0)) == 2


def test_a_resumed_run_continues_its_version_count() -> None:
    """Staleness stays comparable across a restart, so the driver seeds the count it left off at."""
    assert seeded(version=41).publish(params(2.0)) == 42


def test_a_reader_in_this_process_is_never_told_there_is_nothing_new() -> None:
    """Nothing can land while a caller here waits, so there is nothing to wait for and no None to
    return. A Mirror across the cluster is where a reader can genuinely find nothing."""
    store = seeded()

    assert store.get(block=False) is not None
    assert store.get(block=True).version == 0
