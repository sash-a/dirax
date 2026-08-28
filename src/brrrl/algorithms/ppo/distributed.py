from functools import partial
from typing import Any, cast

import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import numpy as np
import optax
import ray
from echo import TrajectoryAccumulator
from gymnasium.wrappers.vector import NormalizeReward

from brrrl.acting.env_stepper import LoopedStepper, prefix
from brrrl.algorithms.ppo.ppo import PPOUpdateRule
from brrrl.algorithms.ppo.types import PPOApply, PPOPolicyOutput, PPOTrajectory, PPOTransition, PPOUpdateState, example
from brrrl.distributed.components.echo import ReplayServer
from brrrl.distributed.components.params import RayParamSink, RayParamStore
from brrrl.distributed.components.trainer import Trainer
from brrrl.distributed.components.update_rule import DataParallelUpdateRule
from brrrl.networks.head import ContinuousHead, DiscreteHead, MaskedDiscreteHead, ValueHead
from brrrl.networks.torso import MLPTorso
from brrrl.params.params import ParamStore
from brrrl.types import VersionedParams
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

    def __call__(self, obs: Any) -> tuple[distrax.Distribution, jax.Array]:
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


def record(obs, term, trunc, reward, policy_output: PPOPolicyOutput):
    return PPOTransition(obs, term, trunc, reward, policy_output.action, policy_output.value, policy_output.log_prob)


def main(config):
    rollout_len = 128
    num_envs = 8
    num_agents = 1
    num_updates = 5000
    num_epochs = 2
    num_minibatches = 4
    gamma = 0.99
    learning_rate = 3e-4
    table_size = 2
    batch_size = 512  # per device
    n_prefetch = 1
    shard_axis_name = "devices"
    env_name = "Acrobot-v1"

    key = jax.random.key(1)
    weights_key, _policy_key, update_key = jax.random.split(key, 3)

    make_env = lambda: rescaled(gym.make(env_name))
    env = make_env()
    stepper = LoopedStepper([make_env], wrap=partial(NormalizeReward, gamma=gamma))
    act_space = one_agent(env.action_space, num_agents)
    width = prefix(stepper.num_envs, num_agents)
    example_transition = example(rollout_len, width, env.observation_space, act_space)
    _traj_accumulator = TrajectoryAccumulator(example_transition)

    network = make_network(env.observation_space, act_space)
    apply_fn = cast(PPOApply, network.apply)
    params = network.init(weights_key, env.observation_space.sample())
    # TODO(params):
    #  Sink on learner -> in learner code
    #  Source per actor -> in actor code
    #  Mirror per actor node -> in actor code
    param_store = (
        # TODO: resources
        ray.remote(RayParamStore).options(**RayParamStore.RAY_OPTIONS).remote(ParamStore(VersionedParams(params, 0)))
    )

    def make_sink():
        return RayParamSink(param_store)

    # --- Actor ---
    # TODO(actor):
    #  placement
    #  inference server
    #  colocation
    #  healthcheck + restarting
    #  must come after learner as they need the IP
    # actors = [
    #     Actor(
    #         actor_config,  # rollout len, num_envs, num actors, placement group??
    #         make_stepper,
    #         make_policy,
    #         record,
    #         PPOTrajectory,
    #     )
    # ]
    # actor_futures = [actor.run() for actor in actors]

    # --- Learner ---
    # TODO(update):
    schedule = optax.linear_schedule(learning_rate, 0.0, num_updates * num_epochs * num_minibatches)
    opt = optax.chain(mean_by_axis(shard_axis_name), optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = opt.init(params)

    # TODO:
    # policy = LocalPolicy(jax.jit(actor_forward), param_source, _policy_key)
    # roller = Roller(stepper, policy, _traj_accumulator, record, PPOTrajectory, rollout_len)
    rule = PPOUpdateRule(apply_fn, opt, num_envs, rollout_len, num_epochs, num_minibatches, gamma=gamma)
    if not rule.isvalid():
        raise ValueError("Invalid config option selected for PPOUpdateRule")
    update_state = PPOUpdateState(params, opt_state, update_key)
    distributed_rule = DataParallelUpdateRule(rule, shard_axis_name)  # TODO: fix typing around rule

    # TODO: example returns a dict currently...
    replay = ReplayServer(table_size, batch_size, n_prefetch, PPOTrajectory(**example_transition))
    recorder = ...  # TODO: distributed/logger.py

    # TODO(learner):
    #  placement
    #  colocation
    #  ray.remote...
    trainer = Trainer(
        num_updates,
        update_state,
        distributed_rule,
        replay,
        make_sink,
        recorder,
    )
    trainer_future = trainer.run()

    ray.wait([trainer_future] + actor_futures)

    # TODO(other):
    #  echo
    #  checkpointing
    #  logging
