"""The hop an actor's Trajectory makes into an update, and the layout it has to land in.

`ReplayServer` is echo's placement rather than a buffer of its own, so what is worth holding it to
is the shape of what comes out of it, that a batch survives long enough to be updated on, and that
a configuration echo cannot serve is refused here rather than hung on.

Two ways in are exercised deliberately. Most of the file feeds a real `echo.Server` in process with
``submit``, which is where a layout claim is cheap to write down; one test puts a real `TcpClient`
on the other end, which is the only thing that can show that an Actor and a Trainer agree about the
pytree crossing between them. `ReplayServer.dataset` takes the servers it reads from as an argument,
which is what lets the two be separate questions.

Every rollout carries its own number in every slot — slot ``t`` of env ``e`` holds ``tag + t + e/10``
— so a batch that stacked the rollouts wrongly, transposed time against envs, or handed back a
recycled ring slot reads differently here rather than merely having a different shape. That is the
failure mode the layout has: all of them are correctly shaped.

The Transition and Trajectory are test-local, which additionally shows that nothing in this seam is
PPO-shaped.
"""

from collections.abc import Iterator
from contextlib import closing
from itertools import product
from typing import Any, NamedTuple

import jax
import numpy as np
import pytest
from conftest import settled
from echo import Server, TcpClient
from jax.sharding import Mesh, PartitionSpec
from numpy.typing import NDArray

from brrrl.algorithms.ppo.ppo import batch_major
from brrrl.distributed.components.echo import ReplayServer, _threaded_prefetch, echo_example
from brrrl.types import Rollout, TrainerBatch

STEPS, ENVS, ROWS = 3, 2, 2
SLOTS = STEPS + 1
"""One slot more than steps taken: a rollout carries the Final obs it stopped on (ADR-0001)."""

TABLE, PREFETCH = 1, 1
BUFFERS = max(2, TABLE + 1)
"""What `ReplayServer` asks echo for, restated so the recycling test can count batches to it."""

TAGS = [100.0, 200.0]
VERSIONS = [4.0, 7.0]
AXIS = "devices"

HEALTH = {
    "replay_store_size",
    "replay_active_connections",
    "replay_queue_depth_sum",
    "replay_queue_depth_max",
    "replay_push_blocked_count",
    "replay_samples_inserted_total",
}


class Transition(NamedTuple):
    observation: Any
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    reward: NDArray[np.floating]
    action: NDArray


class Trajectory(NamedTuple):
    transitions: Transition


def slots(tag: float) -> NDArray[np.float32]:
    """``tag + t + e/10``: every slot of every env distinguishable from every other by value."""
    return np.array([[tag + step + env / 10 for env in range(ENVS)] for step in range(SLOTS)], np.float32)


def trajectory(tag: float) -> Trajectory:
    """The observation takes a trailing axis of its own, so a nested leaf is stacked too."""
    filled = slots(tag)
    return Trajectory(
        Transition(
            observation=filled[..., None],
            terminated=np.zeros((SLOTS, ENVS), np.bool_),
            truncated=np.zeros((SLOTS, ENVS), np.bool_),
            reward=filled,
            action=filled.astype(np.int32),
        )
    )


def blank() -> Trajectory:
    """The example both ends are built from. Its values never travel; only its shapes and dtypes do."""
    return trajectory(0.0)


def replay(prefetch: int = PREFETCH, port: list[int] | None = None) -> ReplayServer[Trajectory]:
    return ReplayServer(TABLE, ROWS, prefetch, blank(), port=port)


def in_process(buffers: int = BUFFERS) -> Server:
    """A Server with no transport at all, so ``submit`` is the only way in and nothing binds."""
    return Server(example=echo_example(blank()), batch_size=ROWS, num_buffers=buffers)


def one_device() -> jax.NamedSharding:
    """A single-device data sharding, which is what a dataset over a single server needs to match."""
    return jax.NamedSharding(Mesh(jax.local_devices()[:1], (AXIS,)), PartitionSpec(AXIS))


def submit(server: Server, tags: list[float], versions: list[float] | None = None) -> None:
    """One batch worth of rollouts. Fed a batch at a time, because echo's ring is what holds them
    and a producer that ran ahead of the consumer by more than it would block rather than buffer."""
    stamps = VERSIONS if versions is None else versions
    for tag, version in zip(tags, stamps, strict=True):
        server.submit(Rollout(trajectory(tag), np.array([version], np.float32)))


def first(server: Server, data: Iterator[TrainerBatch]) -> TrainerBatch:
    submit(server, TAGS)
    return next(data)


@pytest.fixture
def devices() -> list[jax.Device]:
    return jax.local_devices()


@pytest.fixture
def server() -> Iterator[Server]:
    with closing(in_process()) as running:
        yield running


@pytest.fixture
def batch(server: Server) -> TrainerBatch:
    return first(server, replay().dataset([server], one_device()))


def test_one_started_server_stands_behind_each_learner_device(devices: list[jax.Device]) -> None:
    """One buffer per device rather than one shared: each device pulls its own sub-batch, so a
    server short of the mesh is a device that would never be fed."""
    servers = replay().servers(devices)

    try:
        assert len(servers) == len(devices)
        assert len({server.port for server in servers}) == len(devices)
        assert all(server.port > 0 for server in servers)  # a port is only assigned once bound
    finally:
        for server in servers:
            server.close()


def test_ports_that_do_not_match_the_devices_are_refused_before_anything_binds(
    devices: list[jax.Device],
) -> None:
    """Naming ports at all is for a cluster whose firewall was told about them, where landing on the
    wrong count is a run that comes up and then cannot be connected to."""
    mismatched = [0] * (len(devices) + 1)

    with pytest.raises(ValueError) as failure:
        replay(port=mismatched).servers(devices)

    assert str(len(mismatched)) in str(failure.value)
    assert str(len(devices)) in str(failure.value)


def test_a_cpu_learner_gets_its_servers_though_pinned_host_memory_was_asked_for(
    devices: list[jax.Device],
) -> None:
    """Pinning is a CUDA page-lock, and echo raises where it cannot take one. It is asked for by
    default because a GPU learner always wants it, which is exactly what makes it worth a test: no
    caller of a CPU learner ever thinks to turn it off."""
    if any(device.platform == "gpu" for device in devices):
        pytest.skip("the ask is honoured on a GPU learner, which is not what this is about")

    assert replay().pin_host_memory
    servers = replay().servers(devices)

    for server in servers:
        server.close()


def test_a_batch_is_one_row_per_rollout_with_the_rollouts_own_axes_beneath_it(batch: TrainerBatch) -> None:
    """``(R, T+1, E, ...)``. The rollout axis is new and leads; nothing an actor produced is reshaped
    on the way in, which is what lets an algorithm decide for itself how to merge it."""
    transitions = batch.trajectory.transitions

    assert transitions.reward.shape == (ROWS, SLOTS, ENVS)
    assert transitions.terminated.shape == transitions.truncated.shape == (ROWS, SLOTS, ENVS)
    assert transitions.observation.shape == (ROWS, SLOTS, ENVS, 1)


def test_each_row_of_a_batch_is_one_rollout_whole(batch: TrainerBatch) -> None:
    """A stack that interleaved the rollouts is the same shape as one that did not, so this is about
    the numbers: row ``r`` holds rollout ``r``'s own slots, in the order it produced them."""
    reward = batch.trajectory.transitions.reward

    for row, tag in enumerate(TAGS):
        np.testing.assert_array_equal(reward[row], slots(tag))


def test_every_rollout_brings_the_version_it_was_acted_under(batch: TrainerBatch) -> None:
    """Provenance rides beside the payload rather than inside it, one entry per rollout, so a Trainer
    reads the spread of staleness across a batch without a stamp ever reaching a device."""
    assert len(batch.version) == ROWS
    assert batch.version.ravel().tolist() == VERSIONS


def test_a_batch_carries_the_replay_health_a_learner_logs_beside_its_own_metrics(batch: TrainerBatch) -> None:
    """How the buffer and the connections feeding it are holding up is only observable here, on the
    batch. An actor that has stopped sending is a queue depth going to zero and nothing else."""
    assert set(batch.metrics) == HEALTH
    assert batch.metrics["replay_active_connections"] == 0  # this server's own, and nothing dialled it


def test_a_cpu_learners_batch_outlives_the_ring_slot_it_came_from(server: Server) -> None:
    """XLA:CPU reads a host argument when the computation runs rather than when it is dispatched, so
    a batch that stayed a view into echo's ring would be recycled under an update still in flight —
    and the update would succeed, on somebody else's rollouts."""
    data = replay().dataset([server], one_device())
    held = first(server, data).trajectory.transitions.reward
    kept = held.copy()

    for recycle in range(BUFFERS):  # enough batches after it for the ring to come back round
        submit(server, [1000.0 * (recycle + 1)] * ROWS)
        next(data)

    np.testing.assert_array_equal(held, kept)


def test_prefetch_off_is_refused_where_there_is_more_than_one_server_to_gather_from() -> None:
    """A per-device sub-batch has to be gathered from every server before any of it is put, and the
    prefetch thread is where that happens. Without it the second device is simply never read."""
    with closing(in_process()) as one, closing(in_process()) as other, pytest.raises(ValueError) as failure:
        replay(prefetch=0).dataset([one, other], one_device())

    assert "prefetch" in str(failure.value)


def test_a_dataset_needs_one_server_for_every_device_it_will_feed(devices: list[jax.Device]) -> None:
    """Caught where the pair is first put together, rather than as a shard missing from a batch."""
    if len(devices) == 1:
        pytest.skip("one server is the right number for one device")

    sharding = jax.NamedSharding(Mesh(devices, (AXIS,)), PartitionSpec(AXIS))
    with closing(in_process()) as alone, pytest.raises(AssertionError):
        replay().dataset([alone], sharding)


def test_a_rollout_sent_over_the_wire_arrives_as_the_pytree_a_learner_reads(devices: list[jax.Device]) -> None:
    """The only test here with a real client on the other end. An Actor and a Trainer agree about
    what crosses between them only by both calling `echo_example` on the same example trajectory,
    and nothing inside one process can check that they did."""
    servers = replay().servers(devices[:1])
    data = replay().dataset(servers, one_device())

    # `TcpClient` types its pytree as `ndarray`, so a `Rollout` of one does not match — the same
    # accepted diagnostic `actor.py` and the driver carry.
    example = echo_example(blank())
    try:
        with closing(TcpClient("127.0.0.1", servers[0].port, example, max_inflight_msgs=2)) as client:  # ty: ignore[invalid-argument-type]
            for tag, version in zip(TAGS, VERSIONS, strict=True):
                client.send(Rollout(trajectory(tag), np.array([version], np.float32)))  # ty: ignore[invalid-argument-type]
            arrived = next(data)
    finally:
        for server in servers:
            server.close()

    np.testing.assert_array_equal(arrived.trajectory.transitions.reward, [slots(tag) for tag in TAGS])
    assert arrived.version.ravel().tolist() == VERSIONS


def test_the_rollout_axis_merges_out_keeping_every_rollout_and_env_one_row_through_time(
    batch: TrainerBatch,
) -> None:
    """What an algorithm does with the layout above, and the one hop where a wrong answer is
    correctly shaped. ``(R, T+1, E) -> (R·E, T+1)`` has to leave each ``(rollout, env)`` pair a row
    of its own: a merge that took the two axes in the other order produces a batch of exactly the
    right shape, every row of which is a walk across environments rather than through time.

    `brrrl.algorithms.ppo.ppo.batch_major` is the algorithm's side of this seam rather than the
    component's, and it is here because a layout claim needs both ends of it to mean anything.
    """
    merged = batch_major(batch.trajectory)

    assert merged.transitions.reward.shape == (ROWS * ENVS, SLOTS)
    for row, (tag, env) in enumerate(product(TAGS, range(ENVS))):
        np.testing.assert_array_equal(merged.transitions.reward[row], slots(tag)[:, env])


def test_a_prefetched_iterator_yields_what_it_was_given_in_the_order_it_was_given() -> None:
    """The gather it wraps is ordered — each batch is one wave across the devices — so the thread
    between the learner and it must not be free to reorder them."""
    assert list(_threaded_prefetch(iter(range(5)), 2, "test")) == [0, 1, 2, 3, 4]


def test_an_upstream_failure_reaches_the_consumer_rather_than_dying_with_the_thread() -> None:
    """A dead echo server raising inside the prefetch thread would otherwise surface as a learner
    that stopped receiving batches, with the reason it stopped printed on a thread nobody joins."""

    def failing() -> Iterator[int]:
        yield 1
        raise RuntimeError("the server went away")

    items = _threaded_prefetch(failing(), 2, "test")

    assert next(items) == 1
    with pytest.raises(RuntimeError, match="the server went away"):
        next(items)


def test_a_prefetcher_runs_no_further_ahead_than_it_was_told_to() -> None:
    """Each item alive is a batch of device memory held, so the bound is what stops a prefetch of
    one from quietly becoming a prefetch of everything the upstream will produce."""
    produced: list[int] = []

    def upstream() -> Iterator[int]:
        for item in range(100):
            produced.append(item)
            yield item

    items = _threaded_prefetch(upstream(), 2, "test")
    assert next(items) == 0

    settled(lambda: len(produced) >= 2)
    assert produced == [0, 1]  # the consumer holds one and has not come back, so the third is unread
