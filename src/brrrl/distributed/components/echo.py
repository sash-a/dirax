import queue as _queue
import threading
from collections.abc import Iterator
from typing import cast

import jax
import optree
from echo import Sample, Server, TcpTransport

from brrrl.types import Trajectory


def _threaded_prefetch(iterator: Iterator[Sample], size: int, thread_name: str) -> Iterator[Sample]:
    """Runs ``iterator`` in a background thread, buffering up to ``size`` items.

    Exceptions from the upstream iterator are re-raised on the consumer side in the order they were produced.

    Args:
      iterator: The upstream iterator to drain in a background thread.
      size: Max items alive at once, counting the one the consumer holds. Item ``size + 1`` is not
        pulled until the consumer comes back for its next item.
      thread_name: Name for the daemon thread (visible in ``top`` / ``py-spy``).

    Yields:
      Items produced by ``iterator``, in order.
    """
    assert size >= 1, "size must be >= 1"
    buffer: _queue.Queue = _queue.Queue()
    slots = threading.Semaphore(size)
    SENTINEL = object()

    def _producer() -> None:
        try:
            while True:
                slots.acquire()
                buffer.put(next(iterator))
        except StopIteration:
            pass
        # Anything the upstream raises is handed to the consumer rather than lost with this thread.
        except BaseException as exc:  # noqa: BLE001
            buffer.put(exc)
        finally:
            buffer.put(SENTINEL)

    threading.Thread(target=_producer, daemon=True, name=thread_name).start()

    while True:
        item = buffer.get()
        if item is SENTINEL:
            return
        if isinstance(item, BaseException):
            raise item
        try:
            yield item
        finally:
            slots.release()


def _assemble_global_sharded_batch[T: Trajectory](shards: list[T], sharding: jax.sharding.Sharding) -> T:
    """Stitch per-device single-device batch shards into one global sharded Batch.

    Each element of ``shards`` is a Batch pytree already placed on one device. Each shard lands by
    its own device rather than by its position, and its on-device buffer is referenced, not copied.
    """

    def to_global(*device_leaves: jax.Array) -> jax.Array:
        per_device_shape = device_leaves[0].shape
        global_shape = (per_device_shape[0] * len(device_leaves), *per_device_shape[1:])
        return jax.make_array_from_single_device_arrays(global_shape, sharding, list(device_leaves))

    return cast(T, optree.tree_map(to_global, *shards))  # ty: ignore[invalid-argument-type]


def make_echo_servers[T: Trajectory](
    table_size: int,
    per_device_batch_size: int,
    n_devices: int,
    example: T,
    pin_host_memory: bool = True,
    transport_threads: int = 8,
    drainer_threads: int = 8,
    intermediate_queue_size: int = 4,
    port: int = 0,
) -> list[Server]:
    """Create ``n_devices`` echo servers. Each server emits a batch of size ``batch_size`` one for each device.

    Args:
        table_size: the size of each replay buffer as a multiple of batch_size.
        batch_size: the size of the output batch for each device.
        n_devices: the number of devices.
        example: an example trajectory as input by an actor.
        pin_host_memory: whether to pin the echo buffers so D2H transfers are zero-copy.
        transport_threads: the number of threads used to place incoming data from actors onto an intermediate queue.
        drainer_threads: the number of threads to drain the intermediate queue into the buffer.
        intermediate_queue_size: the size of the queue where data waits before it is placed into the buffer.
        port: communication port used for TCP.

    Returns: a list of ``echo.Server``.
    """
    servers: list[Server] = []
    for i in range(n_devices):
        server = Server(
            example=example,
            batch_size=per_device_batch_size,
            transport=TcpTransport(port=port + i, num_threads=transport_threads),
            num_buffers=max(2, int(table_size) + 1),
            num_drainers=drainer_threads,
            producer_queue_size=intermediate_queue_size,
            pin_host_memory=pin_host_memory,
        )
        server.start()
        servers.append(server)
    return servers


def make_echo_iterator(
    servers: list[Server],
    n_prefetch: int,
    sharding: jax.sharding.NamedSharding,
) -> Iterator[Sample]:
    """The single echo iterator for all learner configurations.

    Handles the following cases:
      * CPU learner: one server, and echo copies each batch so its ring slot is freed. XLA:CPU reads
        a host argument when the computation runs rather than when it is dispatched, so an uncopied
        view would be recycled under an in-flight update.
      * n_prefetch <= 0: raw zero-copy iterator over the one server. The H2D transfer is issued
        during dispatch, so the view outlives every read of it. Multi-device needs prefetch for the
        per-device gather.
      * GPU: each device pulls its own server (copy=False, zero-copy) and device_puts that batch
        async, then blocks once on the assembled global sharded batch so all H2D transfers run in
        parallel.
    """
    if n_prefetch <= 0 and len(servers) > 1:
        raise ValueError(
            "prefetch_size <= 0 requires exactly one echo server (<=1 device) multi-server "
            "needs prefetch to gather a per-device sub-batch from each server"
        )

    # TODO: needs to be made to work across multiple nodes
    devices = list(sharding.mesh.devices.flat)
    assert len(servers) == len(devices), f"Need one server per device, got: {len(servers)=}, {len(devices)=}"

    # Echo's own ring buffers are the prefetch on these paths, so no `_threaded_prefetch` thread.
    if all(d.platform == "cpu" for d in devices):
        return servers[0].dataset_iter(copy=True)
    if n_prefetch <= 0:
        return servers[0].dataset_iter(copy=False)

    server_iters = [s.dataset_iter(copy=False) for s in servers]

    def _gpu_iterator() -> Iterator[Sample]:
        while True:
            shards: list[Trajectory] = []
            for device, server_iter in zip(devices, server_iters, strict=True):
                try:
                    sample = next(server_iter)
                except StopIteration:
                    return
                # TODO: only getting info from a single buffer
                info = sample.info
                shards.append(jax.device_put(sample.batch, device))  # async

            batch = _assemble_global_sharded_batch(shards, sharding)
            jax.block_until_ready(batch)  # one block for all parallel transfers
            yield Sample(batch=batch, info=info)

    return _threaded_prefetch(_gpu_iterator(), n_prefetch, thread_name="echo-prefetch")
