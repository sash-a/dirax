"""A real algorithm through the distributed seam, which the components' own tests cannot reach.

Everything either side of this file is held to its contract over a trajectory invented for the
purpose, and rightly: a component that only worked for PPO would be the wrong component. But three
separate compile failures on the distributed path were about what PPO in particular does inside the
mapped function — a `jax.lax.scan` over epochs, an optax state travelling with the parameters, loss
metrics reduced to rank zero — and none of them is reachable from a rule small enough to reason
about. This is the file that runs the real one.

The trap it exists for is `PPOUpdateRule.batch_size`. Under a `shard_map` the rule is handed one
shard, so the size it is built with is the per-device batch and not the run's — and the two are the
same number on the single device the configuration was written on.
`brrrl.algorithms.ppo.distributed` passes rollouts-per-device times envs, and the last test is why.
"""

from typing import cast

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from conftest import replicated
from echo import Server
from jax.sharding import Mesh, PartitionSpec

from brrrl.acting.env_stepper import prefix
from brrrl.algorithms.ppo.ppo import ADAM_EPS, MAX_GRAD_NORM, PPOUpdateRule, make_network
from brrrl.algorithms.ppo.types import PPOApply, PPOTrajectory, PPOTransition, PPOUpdateState, example
from brrrl.distributed.components.echo import ReplayServer, echo_example
from brrrl.distributed.components.update_rule import DataParallelUpdateRule
from brrrl.types import Params, PRNGKey, Rollout
from brrrl.utils.optax import mean_by_axis

AXIS = "devices"
ROLLOUT_LEN, NUM_ENVS, PER_DEVICE = 4, 2, 2
SLOTS = ROLLOUT_LEN + 1
OBS, ACTIONS = 4, 3
EPOCHS, MINIBATCHES = 1, 2

OBS_SPACE = gym.spaces.Box(-1.0, 1.0, (OBS,), np.float32)
ACT_SPACE = gym.spaces.Discrete(ACTIONS)


@pytest.fixture(scope="module")
def network() -> tuple[PPOApply, Params]:
    assembly = make_network(OBS_SPACE, ACT_SPACE)
    return cast(PPOApply, assembly.apply), assembly.init(jax.random.key(0), OBS_SPACE.sample())


def optimizer() -> optax.GradientTransformation:
    """The chain `brrrl.algorithms.ppo.distributed` builds, `mean_by_axis` first."""
    return optax.chain(mean_by_axis(AXIS), optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(1e-3, eps=ADAM_EPS))


def init_params(key: PRNGKey) -> Params:
    """What a rule takes in place of an obs space: an Assembly paired with an example to trace it
    on. Built fresh rather than taken from the fixture, which the same key makes the same params as.
    """
    return make_network(OBS_SPACE, ACT_SPACE).init(key, OBS_SPACE.sample())


def rule(apply_fn: PPOApply, rollouts: int) -> PPOUpdateRule:
    """``rollouts`` is per device: what one shard of the batch holds once it has been merged."""
    return PPOUpdateRule(apply_fn, init_params, optimizer(), rollouts * NUM_ENVS, ROLLOUT_LEN, EPOCHS, MINIBATCHES)


def blank() -> PPOTrajectory:
    """One rollout's buffers, built the way `brrrl.algorithms.ppo.distributed` builds them."""
    return PPOTrajectory(**example(ROLLOUT_LEN, prefix(NUM_ENVS, 1), OBS_SPACE, ACT_SPACE))


def acted(network: tuple[PPOApply, Params], rollouts: int) -> PPOTrajectory:
    """A batch of rollouts as replay hands it over: ``(R, T+1, E, ...)``, values and log
    probabilities the network's own, so the ratio starts at one and the loss is a real number.

    Cast to the dtypes `example` declared, which is what a `TrajectoryAccumulator` does on a run:
    a Discrete action space samples an ``int64`` while a Head under x64-off emits ``int32``, and
    echo takes a caller at its word about widths — it agrees them once, at the handshake.
    """
    apply_fn, params = network
    key = jax.random.key(1)
    width = (rollouts, SLOTS, NUM_ENVS)
    observation = np.asarray(jax.random.uniform(key, (*width, OBS), jnp.float32, -1.0, 1.0))
    distribution, value = apply_fn(params, observation)
    action = distribution.sample(seed=key)
    sampled = PPOTrajectory(
        PPOTransition(
            observation=observation,
            terminated=np.zeros(width, np.bool_),
            truncated=np.zeros(width, np.bool_),
            reward=np.asarray(jax.random.normal(key, width, jnp.float32)),
            action=action,
            value=value,
            log_prob=jnp.asarray(distribution.log_prob(action)),
        )
    )
    return jax.tree.map(lambda leaf, slot: np.asarray(leaf, slot.dtype), sampled, blank())


def test_a_ppo_trajectory_survives_the_round_trip_through_replay(network: tuple[PPOApply, Params]) -> None:
    """PPO's Trajectory is the one that actually crosses on a run, and it is the one with a nested
    observation and an action whose dtype comes off the action space. What echo is given is a
    flattened set of shapes and dtypes, so a field that does not survive is a field silently read
    back as something else."""
    sent = acted(network, PER_DEVICE)
    server = Server(example=echo_example(blank()), batch_size=1, num_buffers=2)
    replay = ReplayServer(1, 1, 1, blank())
    data = replay.dataset([server], jax.NamedSharding(Mesh(jax.local_devices()[:1], (AXIS,)), PartitionSpec(AXIS)))

    try:
        server.submit(Rollout(jax.tree.map(lambda leaf: np.asarray(leaf[0]), sent), np.array([3.0], np.float32)))
        arrived = next(data)
    finally:
        server.close()

    assert jax.tree.structure(arrived.trajectory) == jax.tree.structure(sent)
    for came, went in zip(jax.tree.leaves(arrived.trajectory), jax.tree.leaves(sent), strict=True):
        np.testing.assert_array_equal(came[0], np.asarray(went[0]))


def test_ppos_own_update_rule_compiles_and_runs_under_a_data_parallel_mesh(
    network: tuple[PPOApply, Params],
) -> None:
    """The whole point of the file. A scan over epochs, an optax state carried beside the parameters
    and a dozen loss metrics all have to survive being mapped, and every way they have failed to has
    failed at compile time — at the top of a training run, after everything else was placed."""
    apply_fn, params = network
    rollouts = PER_DEVICE * jax.local_device_count()
    distributed = DataParallelUpdateRule(rule(apply_fn, PER_DEVICE), AXIS)
    mesh, model, data = distributed.shardings()
    state = PPOUpdateState(params, optimizer().init(params), jax.random.key(2))

    updated, metrics = distributed.compiled(mesh, model, data)(
        jax.device_put(state, model), jax.device_put(acted(network, rollouts), data)
    )

    moved = jax.tree.map(lambda before, after: bool(jnp.any(before != after)), params, updated.params)
    assert any(jax.tree.leaves(moved)), "the update ran but left every parameter where it was"
    assert all(np.shape(value) == () and np.isfinite(value) for value in metrics.values())


def test_the_whole_learner_state_lands_replicated_and_not_only_its_parameters(
    network: tuple[PPOApply, Params],
) -> None:
    """`Trainer` builds the state, not the parameters, and PPO's is mostly not parameters: an adam
    chain's step count and its two moment trees ride along, and so does the PRNG key the epochs are
    shuffled with. The update is compiled expecting every one of them replicated, and a key that
    had landed split would shuffle each shard's minibatches differently.

    Through the rule's own `init` rather than a state placed after the fact: building into the
    sharding is the whole reason that call takes one, and a typed key is the leaf that reads
    differently across the two."""
    apply_fn, _ = network
    distributed = DataParallelUpdateRule(rule(apply_fn, PER_DEVICE), AXIS)
    _, model, _ = distributed.shardings()

    assert replicated(distributed.init(jax.random.key(2), model), over=jax.local_device_count())


def test_the_batch_size_a_rule_declares_is_the_shard_it_will_see_and_not_the_whole_batch(
    network: tuple[PPOApply, Params],
) -> None:
    """A rule built for the run's whole batch is handed a shard of it, and says so at the first
    update rather than training on a quarter of the data. Worth pinning because the two are the same
    number on the single device the configuration will have been written on."""
    if jax.local_device_count() == 1:
        pytest.skip("a shard of one device is the whole batch")

    apply_fn, params = network
    rollouts = PER_DEVICE * jax.local_device_count()
    distributed = DataParallelUpdateRule(rule(apply_fn, rollouts), AXIS)
    mesh, model, data = distributed.shardings()
    state = PPOUpdateState(params, optimizer().init(params), jax.random.key(2))

    with pytest.raises(AssertionError, match="shape prefix"):
        distributed.compiled(mesh, model, data)(
            jax.device_put(state, model), jax.device_put(acted(network, rollouts), data)
        )
