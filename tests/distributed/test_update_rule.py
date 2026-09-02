"""What data parallelism owes an update rule, whatever rule that is and however wide the mesh.

Every claim here was broken at least once while the distributed path was being written, and none
of them broke as a wrong number: a plain ``int`` where a `PartitionSpec` belonged, a `shard_map`
with no mesh to run under, per-device metrics crossing an ``out_spec`` that says replicated. Each
was a compile error at the top of a training run, and each is a line of assertion here.

The rule under test is a one-parameter regression rather than PPO, deliberately. Its gradient is a
plain mean over the rows of the batch, so what a correct all-reduce produces is arithmetic a reader
can do in their head, and a split or a reduction that is wrong is a wrong number rather than a
slightly different loss curve.

`brrrl.utils.optax.mean_by_axis` is in the optimiser chain here for the same reason it is in the
one `brrrl.algorithms.ppo.distributed` builds: `shard_map` hands the rule a gradient already summed
over the mesh, and dividing it by the axis size is what makes a device's step the step it would
have taken over the whole batch. The two are only correct together, so they are tested together.
"""

import threading
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from conftest import replicated
from jax.sharding import Mesh, PartitionSpec
from numpy.typing import ArrayLike

from brrrl.distributed.components.update_rule import DataParallelUpdateRule
from brrrl.types import Metrics, Params, UpdateFn
from brrrl.utils.optax import mean_by_axis

COLLECTION, WEIGHT = "params", "w"
"""Flax nests every parameter under the collection it belongs to, and so does `Params`. `weight`
is what reads the one leaf back out as the single number it is here."""

AXIS = "devices"
LEARNING_RATE = 0.1
ROWS = 8
"""Rows in a batch. Divisible by any device count the suite runs on, so a shard is a whole number."""

AWKWARD = (3, 5)
"""A parameter shape owing the mesh nothing — three rows, and never three devices. What a layer
needs is not a multiple of the cards it lands on, which is the whole reason a model is replicated
rather than split."""


class State(NamedTuple):
    params: Params
    opt_state: optax.OptState


@dataclass(frozen=True)
class Squares:
    """Fits one scalar ``w`` against the mean of ``(row · w)²`` over the batch it is handed.

    ``rows`` and ``leading_row`` are metrics about the shard rather than about the fit: the first
    says how much of the batch this device was given and the second says which part of it. Neither
    is symmetric under a wrong split, which the loss and its gradient both are.
    """

    optimizer: optax.GradientTransformation

    def update(self, state: State, batch: jax.Array) -> tuple[State, Metrics]:
        loss = lambda params: jnp.mean(jnp.square(batch * params[COLLECTION][WEIGHT]))
        grads = jax.grad(loss)(state.params)
        updates, opt_state = self.optimizer.update(grads, state.opt_state, state.params)
        params = cast(Params, optax.apply_updates(state.params, updates))
        return State(params, opt_state), {"rows": jnp.float32(batch.shape[0]), "leading_row": batch[0]}

    def isvalid(self) -> bool:
        return True


def optimizer(all_reduce: bool = True) -> optax.GradientTransformation:
    """``all_reduce=False`` is the chain with `mean_by_axis` left out, which one test is about."""
    scaling = [mean_by_axis(AXIS)] if all_reduce else []
    return optax.chain(*scaling, optax.sgd(LEARNING_RATE))


def rule(all_reduce: bool = True) -> DataParallelUpdateRule[State, Any]:
    return DataParallelUpdateRule(Squares(optimizer(all_reduce)), AXIS)


def start(all_reduce: bool = True) -> State:
    params: Params = {COLLECTION: {WEIGHT: jnp.float32(1.0)}}
    return State(params, optimizer(all_reduce).init(params))


def number(value: ArrayLike) -> float:
    """A metric is an `ArrayLike` by the time a recorder would see it, whatever produced it."""
    return float(np.asarray(value))


def weight(state: State) -> jax.Array:
    return cast(jax.Array, state.params[COLLECTION][WEIGHT])


def batch() -> jax.Array:
    """Every row a different number, so which rows a shard was handed is readable off a metric."""
    return jnp.arange(ROWS, dtype=jnp.float32)


def parameters() -> list[jax.Array]:
    """A rank-0 parameter and a rank-2 one: the range a model sharding is asked to take, and neither
    of them a shape the mesh had any say in."""
    return [jnp.float32(1.0), jnp.arange(int(np.prod(AWKWARD)), dtype=jnp.float32).reshape(AWKWARD)]


def over(devices: list[jax.Device]) -> tuple[Mesh, jax.NamedSharding, jax.NamedSharding]:
    """The shardings `DataParallelUpdateRule.shardings` builds, over a chosen slice of the devices.

    Only for the tests that need a mesh narrower than the machine's, which `shardings` cannot give
    them — it spans every local device by definition, and that is its own claim below.
    """
    mesh = Mesh(devices, (AXIS,))
    return mesh, jax.NamedSharding(mesh, PartitionSpec()), jax.NamedSharding(mesh, PartitionSpec(AXIS))


def updated(
    shardings: tuple[Mesh, jax.NamedSharding, jax.NamedSharding],
    all_reduce: bool = True,
) -> tuple[State, Metrics]:
    mesh, model, data = shardings
    update = rule(all_reduce).compiled(mesh, model, data)
    return update(jax.device_put(start(all_reduce), model), jax.device_put(batch(), data))


@pytest.fixture
def devices() -> list[jax.Device]:
    return jax.local_devices()


def test_the_mesh_spans_every_local_device_with_the_model_replicated_and_the_data_split(
    devices: list[jax.Device],
) -> None:
    """Data parallelism is exactly this pair of specs, and they are `PartitionSpec`s rather than the
    axis positions an earlier attempt passed — `shard_map` takes only the former and says so late."""
    mesh, model, data = rule().shardings()

    assert list(mesh.devices.flat) == devices
    assert mesh.axis_names == (AXIS,)
    assert model.spec == PartitionSpec()
    assert data.spec == PartitionSpec(AXIS)


def test_each_device_is_handed_its_own_contiguous_block_of_the_batch(devices: list[jax.Device]) -> None:
    """A batch broadcast whole to every device trains on the same rows n times over and reports
    metrics that look right; only what each shard leads with tells the two apart."""
    _, metrics = updated(rule().shardings())

    shard = ROWS // len(devices)
    assert number(metrics["rows"]) == shard
    assert number(metrics["leading_row"]) == np.mean(np.arange(len(devices)) * shard)


def test_metrics_come_back_as_one_number_for_the_update_rather_than_one_per_shard(
    devices: list[jax.Device],
) -> None:
    """The reduction has to happen inside the mapped function: a rank-0 output cannot cross a mapped
    ``out_spec``, so there is no meaning them afterwards. Downstream is a collector that logs
    scalars, and a metric still carrying a device axis is what it would silently log the first of."""
    _, metrics = updated(rule().shardings())

    assert [np.shape(value) for value in metrics.values()] == [()] * len(metrics)
    assert all(cast(jax.Array, value).sharding.is_fully_replicated for value in metrics.values())


def test_a_data_parallel_update_is_the_update_one_device_would_have_made_over_the_whole_batch(
    devices: list[jax.Device],
) -> None:
    """The whole promise. Splitting the batch is a throughput decision, and a run that took a
    different step for having been spread over more cards would make every comparison meaningless."""
    spread, _ = updated(rule().shardings())
    alone, _ = updated(over(devices[:1]))

    assert float(weight(spread)) == pytest.approx(float(weight(alone)))


def test_the_step_is_the_one_the_mean_gradient_over_the_whole_batch_asks_for(devices: list[jax.Device]) -> None:
    """The single-device reference above is only worth as much as its own correctness, so the pair
    is anchored to a number written out here rather than to each other."""
    state, _ = updated(rule().shardings())

    gradient = float(np.mean(2 * np.arange(ROWS) ** 2))  # d/dw of mean((row·w)²) at w = 1
    assert float(weight(state)) == pytest.approx(1.0 - LEARNING_RATE * gradient)


def test_without_the_axis_mean_the_gradient_arrives_multiplied_by_the_device_count(
    devices: list[jax.Device],
) -> None:
    """Why `mean_by_axis` is in the chain at all. `shard_map` sums a device-varying gradient on its
    way out, so a chain without it takes a step ``n`` times too large — which on a real run is a
    learning rate silently scaled by however many cards the job happened to get."""
    if len(devices) == 1:
        pytest.skip("a sum over one device is its own mean")

    unscaled, _ = updated(rule().shardings(), all_reduce=False)
    scaled, _ = updated(rule().shardings())

    walked = 1.0 - float(weight(scaled))
    assert 1.0 - float(weight(unscaled)) == pytest.approx(walked * len(devices))


def test_every_device_ends_the_update_holding_the_same_parameters(devices: list[jax.Device]) -> None:
    """Replicated is a claim about the values on each device, not only about the sharding they are
    labelled with. Replicas that had drifted would make every actor's parameters depend on which
    shard answered."""
    state, _ = updated(rule().shardings())

    assert replicated(state.params, over=len(devices))


def test_the_model_sharding_puts_a_whole_copy_of_a_parameter_on_every_device(
    devices: list[jax.Device],
) -> None:
    """Whether `Trainer`'s ``device_put`` against this sharding replicates, which was an open
    question: it does, for any shape, because `PartitionSpec()` names no axis to cut along.

    Asserted by how many devices hold a copy and how much each holds, never by
    ``is_fully_replicated`` — that is True of a `SingleDeviceSharding` too, so a parameter still
    sitting where it was built passes it and a state that was never placed reads as replicated.
    """
    _, model, _ = rule().shardings()

    for value in parameters():
        placed = jax.device_put(value, model)
        assert len(placed.addressable_shards) == len(devices)
        assert [shard.data.shape for shard in placed.addressable_shards] == [value.shape] * len(devices)


def test_no_parameter_would_have_fitted_the_data_sharding_instead(devices: list[jax.Device]) -> None:
    """Why the model and the data get separate shardings rather than one applied to both. A batch
    leads with an axis sized by the run and a parameter leads with whatever its layer needed, so the
    spec that splits a batch cannot take either of these: rank-0 has no axis at all, and three rows
    do not divide by two devices."""
    if len(devices) == 1:
        pytest.skip("a mesh of one splits anything divisible by one, which is every batch there is")

    _, _, data = rule().shardings()

    for value in parameters():
        with pytest.raises(ValueError):
            jax.device_put(value, data)


def test_the_compiled_update_runs_on_a_thread_that_was_never_given_a_mesh(devices: list[jax.Device]) -> None:
    """A Trainer's ``run`` is called on whatever thread Ray hands it, and an ambient mesh is
    thread-local. The mesh is an argument to `shard_map` here for that reason, so nothing about the
    update depends on where it is called from."""
    mesh, model, data = rule().shardings()
    update: UpdateFn = rule().compiled(mesh, model, data)
    state, arrived = jax.device_put(start(), model), jax.device_put(batch(), data)
    elsewhere: list[State] = []

    thread = threading.Thread(target=lambda: elsewhere.append(update(state, arrived)[0]))
    thread.start()
    thread.join()

    assert float(weight(elsewhere[0])) == pytest.approx(float(weight(update(state, arrived)[0])))
