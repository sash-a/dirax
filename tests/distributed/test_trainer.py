"""The learner's loop: pull a batch, update on it, publish the parameters, say what happened.

Everything a Trainer holds is real here except replay, which is scripted so a test can say which
batch arrived under which version and read back what was done with it. The `DataParallelUpdateRule`
is the real one, over a rule small enough that a wrong batch is a wrong number rather than a
slightly different loss; the `ParamStore` is the same in-process Store a Ray-placed one holds
rather than reimplements; and the recorder is the far side of the run's one collector.

The rule folds each batch's mean into a single parameter a decimal place at a time, so the state is
a number that says which batches reached it, in what order — ``1``, ``2`` and ``3`` arriving in
order leave ``123``. A batch delivered twice, dropped, or applied out of order is a different
number rather than a plausible one.

What is not here is anything about how a batch got to the learner or what the update means. Both
have tests of their own; a Trainer is the seam between them, and the seam is what breaks: a batch
built over a different sharding than the update was compiled against, a version subtracted from
the wrong side, a replay server the driver never stopped.
"""

from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import Recorded

from brrrl.distributed.components.trainer import Trainer
from brrrl.distributed.components.update_rule import DataParallelUpdateRule
from brrrl.metrics.collector import WindowedRecorder
from brrrl.params.params import ParamStore
from brrrl.types import Metrics, Params, TrainerBatch, VersionedParams

AXIS = "devices"
ROWS = 4
"""Rollouts in a batch. Divisible by any device count the suite runs on, so a shard is whole."""

MEANS = [1.0, 2.0, 3.0]
"""One per batch, and one decimal place each: the state after all three is ``123``."""

ACTED_UNDER = [[0.0, 0.0], [1.0, 2.0], [0.0, 4.0]]
"""The Versions each batch's rollouts were acted under. Deliberately uneven, so a staleness that
took the first entry, or the trainer's own Version, is not accidentally the same number."""

STALENESS = [2.0, 1.5, 2.0]
"""``publish`` hands back the Version it just assigned. The driver seeded the Store at zero and the
Trainer published what it started from, so the update that consumes batch ``i`` produces ``i + 2``."""

DEPTH = [11, 22, 33]
"""A replay health number that differs per batch, so a learner logging a stale one is visible."""

TIMED = {"traj_wait_time", "update_dispatch_time", "param_publish_time", "checkpointing_time"}
PORT = 40000
SEEN = "seen"
"""The one parameter. `Params` is a nest of mappings to the type system, so `folded` reads it back
as the single number the rule below keeps in it."""


class State(NamedTuple):
    params: Params


@dataclass(frozen=True)
class Decimals:
    """Shifts the parameter a place and adds the batch's mean, so order and arrival are both read
    off the one number. The mean is taken over the whole mesh, which is the update rule's job to
    make possible and not this rule's to check."""

    def update(self, state: State, batch: jax.Array) -> tuple[State, Metrics]:
        mean = jax.lax.pmean(jnp.mean(batch), AXIS)
        return State({SEEN: cast(jax.Array, state.params[SEEN]) * 10 + mean}), {"batch_mean": mean}

    def isvalid(self) -> bool:
        return True


@dataclass
class Stopping:
    """An echo server as a Trainer holds one: a port to advertise, and a close to be called."""

    port: int
    closed: bool = False

    def close(self) -> None:
        self.closed = True


@dataclass
class Scripted:
    """Replay as a Trainer uses it: somewhere to get servers from, and batches to hand back.

    Standing in for `brrrl.distributed.components.echo.ReplayServer` is what lets a test say which
    batch arrived under which Version. That it can stand in at all, holding none of echo, is the
    case `ReplayServer`'s own TODO makes for its being a protocol.
    """

    batches: list[TrainerBatch[Any]]
    built: list[Stopping] = field(default_factory=list)
    shardings: list[jax.NamedSharding] = field(default_factory=list)

    def servers(self, devices: list[jax.Device]) -> list[Stopping]:
        self.built = [Stopping(PORT + index) for index in range(len(devices))]
        return self.built

    def dataset(self, servers: list[Stopping], sharding: jax.NamedSharding) -> Any:
        self.shardings.append(sharding)
        return iter(self.batches)


@dataclass
class Wired:
    """One Trainer and everything it was built over, which is where the assertions are."""

    trainer: Trainer[Any, Any]
    replay: Scripted
    store: ParamStore
    collected: Recorded
    """Where the records land, held apart from the recorder because a window may sit between."""


def batches(health: list[Any] | None = None) -> list[TrainerBatch[Any]]:
    """A batch is a flat array here rather than a Trajectory: what a Trainer does with one is place
    it and hand it to the update, and neither of those reads a field of it."""
    depths = DEPTH if health is None else health
    return [
        TrainerBatch[Any](
            np.full(ROWS, mean, np.float32),
            np.array(versions, np.float32),
            {"replay_queue_depth_max": depth},
        )
        for mean, versions, depth in zip(MEANS, ACTED_UNDER, depths, strict=True)
    ]


def build(feed: list[TrainerBatch[Any]] | None = None, window: int = 0) -> Wired:
    """The driver seeds the Store before placing anything, which is what the Version count is from.

    ``window`` puts a real `WindowedRecorder` between the Trainer and where the records land, which
    is the chain a run builds; zero leaves the Trainer recording straight into it.
    """
    params: Params = {SEEN: jnp.float32(0.0)}
    store = ParamStore(VersionedParams(params, 0))
    replay = Scripted(batches() if feed is None else feed)
    collected = Recorded()
    recorder = WindowedRecorder(collected, window) if window else collected
    rule = DataParallelUpdateRule(Decimals(), AXIS)
    trainer = Trainer(len(replay.batches), State(params), rule, replay, lambda: store, recorder)  # ty: ignore[invalid-argument-type]
    return Wired(trainer, replay, store, collected)


def folded(params: Params) -> float:
    return float(cast(jax.Array, params[SEEN]))


def replicated(params: Params) -> bool:
    """Fully replicated by its sharding, and holding the same values on every device it names."""
    leaves: list[jax.Array] = jax.tree.leaves(params)
    return all(leaf.sharding.is_fully_replicated for leaf in leaves) and all(
        np.array_equal(shard.data, leaf.addressable_shards[0].data)
        for leaf in leaves
        for shard in leaf.addressable_shards
    )


def recorded(wired: Wired) -> list[dict[str, Any]]:
    return [metrics for metrics, _ in wired.collected.records]


def test_a_trainer_publishes_what_it_starts_from_before_it_takes_a_single_batch() -> None:
    """Actors act under whatever the Store holds, and what a Trainer starts from is not always what
    the driver seeded — a resumed run starts from a checkpoint. Publishing it up front is what makes
    the Store's contents the learner's own state rather than the driver's guess at it."""
    wired = build()

    assert wired.store.get(block=False).version == 1
    assert folded(wired.trainer.state.params) == 0.0


def test_a_learner_folds_in_every_batch_it_is_given_in_the_order_they_arrive() -> None:
    """One decimal place per batch: means of 1, 2 and 3 in order leave 123. A batch consumed twice,
    skipped, or reordered against its Version is a number that is not that."""
    wired = build()

    wired.trainer.run()

    assert folded(wired.trainer.state.params) == pytest.approx(123.0)


def test_every_update_publishes_the_parameters_it_produced() -> None:
    """An unpublished update is one no actor will ever act under, and the run's staleness grows
    without anything reporting that it has."""
    wired = build()

    wired.trainer.run()

    published = wired.store.get(block=False)
    assert published.version == 1 + len(MEANS)
    assert folded(published.value) == pytest.approx(123.0)


def test_staleness_is_how_far_behind_the_parameters_a_batch_produced_its_rollouts_were_acted_under() -> None:
    """The one number that says whether the run is still on-policy enough for a clipped ratio to
    mean anything. It is a difference across three processes, so the sign and which Version is on
    which side of it are both easy to get wrong and impossible to spot on a chart."""
    wired = build()

    wired.trainer.run()

    assert [float(metrics["staleness"]) for metrics in recorded(wired)] == STALENESS


def test_a_learners_record_carries_the_health_of_the_batch_it_consumed_beside_its_own_metrics() -> None:
    """Replay health, the update's metrics, staleness and where the loop spent its time all leave in
    one record, so all four are stamped with the same moment in the run."""
    wired = build()

    wired.trainer.run()

    expected = sorted({"replay_queue_depth_max", "batch_mean", "staleness", *TIMED})
    assert wired.collected.keys == [expected] * len(MEANS)
    assert [metrics["replay_queue_depth_max"] for metrics in recorded(wired)] == DEPTH


def test_a_learner_advances_no_step_of_its_own() -> None:
    """Cumulative transitions is the run's only x-axis and only Actors produce them, so a learner's
    metrics are stamped with where the run had got to rather than with a count of its own updates
    (ADR-0005). A learner that passed its update count would drive the clock backwards against the
    actors' and land every loss curve somewhere no return curve is."""
    wired = build()

    wired.trainer.run()

    assert wired.collected.transitions == [0] * len(MEANS)


def test_the_state_a_learner_starts_from_is_replicated_onto_every_device() -> None:
    """``device_put`` against a replicated `NamedSharding` is what puts a whole copy of the model on
    each device, and the update is compiled expecting exactly that. A state that had been split
    instead is a shape error at the first update, and one left on the host is a transfer per step."""
    wired = build()

    assert replicated(wired.trainer.state.params)


def test_the_state_a_learner_carries_forward_is_still_replicated_after_an_update() -> None:
    """Every device has to leave an update agreeing about the model, or which shard answered decides
    what an actor acts under."""
    wired = build()

    wired.trainer.run()

    assert replicated(wired.trainer.state.params)


def test_one_replay_server_is_started_for_each_device_in_the_learners_mesh() -> None:
    """Each device pulls its own sub-batch, so the servers and the mesh are the same width or a
    device goes unfed."""
    wired = build()

    assert len(wired.replay.built) == jax.local_device_count()


def test_the_dataset_is_built_over_the_very_sharding_the_update_was_compiled_against() -> None:
    """The batch arrives already placed. Assembled against a different sharding than the update's
    ``in_spec`` it would be resharded on the way in at best, and land wrong at worst."""
    wired = build()

    _, _, data_sharding = DataParallelUpdateRule(Decimals(), AXIS).shardings()
    assert wired.replay.shardings == [data_sharding]


def test_close_stops_every_replay_server_and_lets_go_of_the_recorder() -> None:
    """Called once every Actor has stopped and never before: the drainer threads behind these
    servers are what acknowledge an Actor's in-flight sends, and an Actor blocked in ``send`` is
    waiting on an acknowledgement rather than on anything it can be told to abandon."""
    wired = build()
    wired.trainer.run()

    wired.trainer.close()

    assert all(server.closed for server in wired.replay.built)
    assert wired.collected.closes == 1


def test_a_window_that_never_filled_still_reaches_the_collector_when_the_trainer_closes() -> None:
    """A learner records once an update, and a run that stops mid-window would otherwise lose every
    metric since the last flush — including the ones from the update it stopped on."""
    wired = build(window=1000)
    wired.trainer.run()
    reached = len(wired.collected.records)

    wired.trainer.close()

    assert reached == 1  # the first record goes straight through, so a run says something at once
    held = wired.collected.records[-1][0]
    assert float(np.asarray(held["staleness/mean"])) == pytest.approx(np.mean(STALENESS[1:]))


def test_a_metric_that_arrives_as_a_list_is_folded_like_any_other() -> None:
    """A learner forwards whatever the batch carried without meaning it first, and a replay metric
    sampled per buffer arrives as a list rather than a scalar. The window folds every value in it,
    so what reaches the backend is one number either way."""
    wired = build(feed=batches([[1.0, 2.0, 3.0]] * len(MEANS)), window=1000)

    wired.trainer.run()
    wired.trainer.close()

    held = wired.collected.records[-1][0]
    assert float(np.asarray(held["replay_queue_depth_max/mean"])) == pytest.approx(2.0)


def test_the_replay_addresses_name_one_port_per_server() -> None:
    """The one thing an Actor needs off the Trainer, and it cannot be known before the servers bind."""
    wired = build()

    ports = [address.rsplit(":", 1)[1] for address in wired.trainer.get_replay_addresses()]

    assert ports == [str(server.port) for server in wired.replay.built]
