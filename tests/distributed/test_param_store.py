"""The store readers park on, and the writer that mints the Versions they read.

What a `RayParamStore` is for is the window nothing else in the run covers: params are built on
the Learner, so between the driver placing the store and the Learner's opening publish the store
holds nothing at all. Every reader has to wait out that window and then see the Learner's first
state exactly once, under the Version the Learner gave it.

Driven as plain coroutines rather than through Ray. The methods are the ones Ray places, and what
is worth asserting about them is the wake each reader gets and what it reads when it is woken,
which is a property of this class rather than of the transport under it.
"""

import asyncio
from collections.abc import Awaitable
from typing import Protocol, cast

import jax.numpy as jnp
import pytest

from brrrl.distributed.components.params import RayParamSink, RayParamStore
from brrrl.types import Params, VersionedParams
from brrrl.utils.duration import EveryTicks

READER, OTHER = 0, 1
SETTLE = 0.05
"""Long enough for a `get` that was going to resolve to have resolved. Nothing races it."""


def params(scale: float) -> Params:
    return {"params": {"scale": jnp.float32(scale)}}


def scale_of(held: VersionedParams) -> float:
    return float(held.value["params"]["scale"])


class InProcess(Protocol):
    """The methods Ray places, as the coroutines they are behind the decorator."""

    async def publish(self, params: VersionedParams) -> None: ...
    async def get(self, reader_id: int) -> VersionedParams: ...


def store() -> InProcess:
    return cast(InProcess, RayParamStore([READER, OTHER]))


class Recording:
    """The store's side of a publish, which is another process on a run and so only visible here."""

    def __init__(self) -> None:
        self.published: list[VersionedParams] = []

    @property
    def publish(self) -> "Recording":
        """A `RayParamSink` calls ``store.publish.remote(...)``, so ``publish`` is the stand-in."""
        return self

    def remote(self, params: VersionedParams) -> None:
        self.published.append(params)


async def settled[T](awaitable: Awaitable[T]) -> T | None:
    """What the awaitable produced, or None where it was still parked after `SETTLE`."""
    task = asyncio.ensure_future(awaitable)
    done, _ = await asyncio.wait([task], timeout=SETTLE)
    return task.result() if done else None


def test_a_reader_parks_until_the_learner_publishes_rather_than_reading_an_empty_store() -> None:
    """The window this store exists to survive: params are built on the Learner, so the driver has
    nothing to seed it with and a reader that arrived first has nothing to read. Waiting is the
    only answer that is not a default nobody trained."""

    async def body() -> None:
        held = store()
        assert await settled(held.get(READER)) is None

    asyncio.run(body())


def test_the_publish_that_ends_that_wait_is_the_one_the_reader_is_handed() -> None:
    """A reader parked since before the run began comes back with the Learner's opening state,
    under the Version the Learner minted for it."""

    async def body() -> None:
        held = store()
        parked = asyncio.ensure_future(held.get(READER))
        await held.publish(VersionedParams(params(1.0), 0))

        first = await parked
        assert (first.version, scale_of(first)) == (0, 1.0)

    asyncio.run(body())


def test_a_reader_behind_several_publishes_reads_the_newest_and_not_the_next_one() -> None:
    """Staleness is the cost this store exists to bound. A reader that steps one publish per read
    never catches up with a Learner publishing faster than it polls."""

    async def body() -> None:
        held = store()
        for version, scale in enumerate([1.0, 2.0, 3.0]):
            await held.publish(VersionedParams(params(scale), version))

        latest = await held.get(READER)
        assert (latest.version, scale_of(latest)) == (2, 3.0)

    asyncio.run(body())


def test_publishes_a_reader_slept_through_wake_it_once_between_them() -> None:
    """The wake is edge-triggered per reader, so three publishes owe a reader one read and not
    three. A second read finds nothing new and parks, which is what keeps a Sync point cheap."""

    async def body() -> None:
        held = store()
        for version in range(3):
            await held.publish(VersionedParams(params(1.0), version))

        await held.get(READER)
        assert await settled(held.get(READER)) is None

    asyncio.run(body())


def test_every_reader_is_woken_by_a_publish_and_not_only_the_first_to_ask() -> None:
    """One wake per reader rather than one per publish: a reader that claimed a publish must not
    have claimed it on behalf of the others."""

    async def body() -> None:
        held = store()
        await held.publish(VersionedParams(params(1.0), 0))
        await held.get(READER)

        assert (await held.get(OTHER)).version == 0

    asyncio.run(body())


def test_a_reader_the_store_was_never_told_about_is_an_error_rather_than_a_silent_wait() -> None:
    """Reader ids are fixed at construction. An id nobody registered would otherwise park forever
    and read as an actor that is merely slow."""

    async def body() -> None:
        held = store()
        with pytest.raises(KeyError):
            await held.get(READER + OTHER + 1)

    asyncio.run(body())


def test_a_fresh_runs_opening_publish_is_version_zero() -> None:
    """The Learner mints Versions, so the count starts where its first publish lands rather than
    where a driver's seed did."""
    sink = RayParamSink(Recording(), EveryTicks(1))  # ty: ignore[invalid-argument-type]

    assert sink.publish(params(1.0)) == 0


def test_a_resumed_run_carries_on_from_the_version_its_checkpoint_left_off_at() -> None:
    """Versions are the run's monotonic authority, and a resume that restarted them at zero would
    make a staleness reading meaningless across it."""
    sink = RayParamSink(Recording(), version=41)  # ty: ignore[invalid-argument-type]

    assert sink.publish(params(1.0)) == 42


def test_a_skipped_publish_costs_no_version_and_reports_the_last_one_that_landed() -> None:
    """The Learner reads the returned Version back as its own, so a publish the schedule declined
    has to report what actors are still acting under rather than a Version that never existed."""
    recording = Recording()
    sink = RayParamSink(recording, EveryTicks(2))  # ty: ignore[invalid-argument-type]

    reported = [sink.publish(params(scale)) for scale in [1.0, 2.0, 3.0, 4.0]]

    assert reported == [0, 0, 1, 1]
    assert [held.version for held in recording.published] == [0, 1]
