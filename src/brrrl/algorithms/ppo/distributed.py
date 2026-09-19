from functools import partial
from typing import Any

import distrax
import echo
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray
from chex import PRNGKey
from echo import TrajectoryAccumulator
from gymnasium.wrappers.vector import NormalizeReward
from ray import ObjectRef
from ray.actor import ActorProxy

from brrrl.acting.env_stepper import LoopedStepper, prefix
from brrrl.acting.policy import FFPolicy
from brrrl.acting.roller import Roller
from brrrl.algorithms.ppo.ppo import ADAM_EPS, MAX_GRAD_NORM, PPOUpdateRule
from brrrl.algorithms.ppo.types import (
    PPOApply,
    PPOPolicyInput,
    PPOPolicyOutput,
    PPOTrajectory,
    PPOTransition,
    PPOUpdateState,
    example,
)
from brrrl.checkpointing.checkpointer import NamedTupleLayout, OrbaxCheckpointer, get_save_path
from brrrl.distributed.components.actor import Actor
from brrrl.distributed.components.checkpointer import CheckpointStore, RayCheckpointer
from brrrl.distributed.components.echo import ReplayServer, echo_example
from brrrl.distributed.components.metrics import RayMetricCollector, default_remote_logger
from brrrl.distributed.components.params import RayParamSink, RayParamSource, RayParamStore
from brrrl.distributed.components.trainer import Trainer
from brrrl.distributed.components.update_rule import DataParallelUpdateRule
from brrrl.distributed.placement.placement import Placement, Placements
from brrrl.distributed.placement.preflight import Placed, preflight
from brrrl.metrics.stdout import StdoutLogger
from brrrl.networks.head import ContinuousHead, DiscreteHead, MaskedDiscreteHead, ValueHead
from brrrl.networks.torso import MLPTorso
from brrrl.types import Params
from brrrl.utils.duration import EveryTicks
from brrrl.utils.optax import mean_by_axis
from brrrl.wrappers.action_mask import ACTION_MASK, OBSERVATION
from brrrl.wrappers.agents import one_agent


# Duplication of the code in ppo.py on purpose. This demonstrates how I'd use this library in
# a distributed setting and makes it easy to use this as a starting point.
class PPONetwork(nn.Module):
    policy: nn.Module

    def setup(self):
        self.torso = MLPTorso([64, 64], activation=jax.nn.relu)
        self.value = ValueHead()

    def __call__(self, policy_input: PPOPolicyInput) -> tuple[distrax.Distribution, jax.Array]:
        obs: Any = policy_input.observation
        masked = isinstance(self.policy, MaskedDiscreteHead)
        emb = self.torso(obs[OBSERVATION] if masked else obs)
        return (self.policy(emb, obs[ACTION_MASK]) if masked else self.policy(emb)), self.value(emb)


def make_network(obs_space: gym.Space, act_space: gym.Space) -> PPONetwork:
    """``act_space`` is one agent's. `brrrl.wrappers.agents.one_agent` of what the env declares."""
    is_masked = isinstance(obs_space, gym.spaces.Dict) and ACTION_MASK in obs_space.spaces
    match act_space:
        case gym.spaces.Discrete() if is_masked:
            return PPONetwork(MaskedDiscreteHead(int(act_space.n)))
        case gym.spaces.Discrete():
            return PPONetwork(DiscreteHead(int(act_space.n)))
        case gym.spaces.Box(shape=(num_actions,)):
            return PPONetwork(ContinuousHead(num_actions))
        case _:
            raise ValueError(f"{act_space} must be discrete or a one-dimensional Box")


def rescaled(env: gym.Env) -> gym.Env:
    """A continuous env stepped in ``(-1, 1)``, which is the interval a `ContinuousHead` samples in."""
    if isinstance(env.action_space, gym.spaces.Box):
        return gym.wrappers.RescaleAction(env, np.float32(-1.0), np.float32(1.0))
    return env


def close(
    actors: list[ActorProxy[Actor]],
    actor_futures: list[ObjectRef],
    trainer: ActorProxy[Trainer],
    collector: ActorProxy[RayMetricCollector],
    checkpointer: RayCheckpointer[PPOUpdateState],
    timeout: float = 60,
) -> None:
    """Unwind outwards from what blocks: the Actors, then the servers they send to, then the collector.

    Raises:
        TimeoutError: An Actor was still running ``timeout`` seconds after being asked to close.
    """
    ray.get([actor.close.remote() for actor in actors])
    checkpointer.close()
    stopped, running = ray.wait(actor_futures, num_returns=len(actor_futures), timeout=timeout)
    ray.get(trainer.close.remote(), timeout=timeout)
    ray.get(collector.close.remote(), timeout=timeout)
    ray.get(stopped)  # An Actor that died mid-run is a real error, not a quiet one.
    if running:
        raise TimeoutError(f"{len(running)}/{len(actors)} actors still running {timeout:g}s after close")


def main():
    rollout_len = 128
    num_envs = 1
    num_actors = 8
    actor_ids = list(range(num_actors))
    num_agents = 1
    num_updates = 5000
    batch_size = 4  # number of rollouts per device
    num_epochs = 2
    num_minibatches = 4
    timesteps_per_device = rollout_len * num_envs * batch_size
    gamma = 0.99
    learning_rate = 3e-4
    table_size = 2
    n_prefetch = 1
    checkpoint_every = EveryTicks(25)
    shard_axis_name = "devices"
    env_name = "Acrobot-v1"

    placements = Placements(trainer=Placement(num_cpus=4, num_gpus=1), actor=Placement(num_cpus=1))

    ray.init()
    key = jax.random.key(1)
    policy_key, trainer_key = jax.random.split(key)

    make_env = lambda: rescaled(gym.make(env_name))
    env = make_env()
    act_space = one_agent(env.action_space, num_agents)
    width = prefix(num_envs, num_agents)
    example_transition = example(rollout_len, width, env.observation_space, act_space)

    network = make_network(env.observation_space, act_space)
    apply_fn: PPOApply = lambda params, policy_input, per_rollout: network.apply(params, policy_input)  # type: ignore

    # --- Learner ---
    # TODO(learner):
    #  colocation
    schedule = optax.linear_schedule(learning_rate, 0.0, num_updates * num_epochs * num_minibatches)
    opt = optax.chain(
        mean_by_axis(shard_axis_name),
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adam(schedule, eps=ADAM_EPS),
    )
    example_input = PPOPolicyInput(env.observation_space.sample())
    init_params = lambda weights_key: network.init(weights_key, example_input)
    rule = PPOUpdateRule(
        apply_fn, init_params, opt, batch_size * num_envs, rollout_len, num_epochs, num_minibatches, gamma=gamma
    )
    if not rule.isvalid():
        raise ValueError("Invalid config option selected for PPOUpdateRule")

    distributed_rule = DataParallelUpdateRule(rule, shard_axis_name)

    replay = ReplayServer(table_size, batch_size, n_prefetch, PPOTrajectory(**example_transition))
    actor_recorder, learner_recorder, collector = default_remote_logger(
        StdoutLogger, EveryTicks(20), placements.metrics
    )

    # TODO(params):
    #  Mirror per actor node -> in actor code
    param_store = placements.params.apply(RayParamStore).remote(actor_ids)

    save_path = get_save_path().resolve()
    checkpoint_store = placements.checkpoints.apply(CheckpointStore).remote(
        lambda: OrbaxCheckpointer(save_path, NamedTupleLayout(PPOUpdateState))
    )
    checkpointer = RayCheckpointer(checkpoint_store, every=checkpoint_every)

    trainer: ActorProxy[Trainer] = placements.trainer.apply(Trainer).remote(
        num_updates,
        timesteps_per_device,
        trainer_key,
        distributed_rule,
        replay,
        RayParamSink(param_store),
        learner_recorder,
        checkpointer,
    )

    # Waited on here because the call below blocks on the Trainer, and an unplaceable one hangs it.
    preflight(
        [
            Placed("trainer", placements.trainer, trainer),
            Placed("params", placements.params, param_store),
            Placed("checkpoints", placements.checkpoints, checkpoint_store),
            Placed("metrics", placements.metrics, collector),
        ]
    )
    replay_addrs = ray.get(trainer.get_replay_addresses.remote())

    # --- Actor ---
    # TODO(actor):
    #  inference server
    #  healthcheck + restarting (actor group)
    def actor_forward(params: Params, policy_input: PPOPolicyInput, key: PRNGKey) -> PPOPolicyOutput:
        distr, value = apply_fn(params, policy_input, ())
        action = distr.sample(seed=key)
        logp = jnp.asarray(distr.log_prob(action))
        return PPOPolicyOutput(action, logp, value)

    def make_roller(actor_id: int, key: PRNGKey) -> Roller:
        iinfo = jnp.iinfo(jnp.int32)
        seed = jax.device_get(jax.random.randint(key, (), 0, iinfo.max, jnp.int32)).item()
        # TODO: this needs a mirror per actor and needs to ref the mirror
        param_source = RayParamSource(param_store, actor_id)
        traj_accumulator = TrajectoryAccumulator(example_transition)
        policy = FFPolicy(jax.jit(actor_forward), param_source, key)
        stepper = LoopedStepper([make_env for _ in range(num_envs)], wrap=partial(NormalizeReward, gamma=gamma))
        return Roller(
            stepper, policy, traj_accumulator, PPOPolicyInput, PPOTransition, PPOTrajectory, rollout_len, seed
        )

    def make_replay_client(i: int) -> echo.TcpClient:
        replay_addr = replay_addrs[i % len(replay_addrs)]
        host, port = replay_addr.split(":")
        example = echo_example(PPOTrajectory(**example_transition))
        return echo.TcpClient(host, int(port), example, max_inflight_msgs=2)  # ty: ignore[invalid-argument-type]

    policy_keys = jax.random.split(policy_key, num_actors)
    actor_class = placements.actor.apply(Actor)
    actors: list[ActorProxy[Actor]] = [
        actor_class.remote(
            partial(make_roller, i, policy_keys[i]),
            partial(make_replay_client, i),
            actor_recorder,
        )
        for i in actor_ids
    ]
    preflight([Placed("actor", placements.actor, actor) for actor in actors])

    actor_futures = [actor.run.remote() for actor in actors]
    try:
        ray.get(trainer.run.remote())
    finally:
        try:
            close(actors, actor_futures, trainer, collector, checkpointer)
        finally:
            ray.shutdown()

    # TODO(other):
    #  checkpointing
    #  eval


if __name__ == "__main__":
    main()
