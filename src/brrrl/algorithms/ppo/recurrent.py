from functools import partial
from typing import Any, final

import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from echo import TrajectoryAccumulator
from gymnasium.wrappers.vector import NormalizeReward
from jax import tree

from brrrl.acting.env_stepper import AsyncStepper, prefix
from brrrl.acting.policy import RecPolicy
from brrrl.acting.roller import Roller
from brrrl.algorithms.ppo.ppo import ADAM_EPS, MAX_GRAD_NORM, TORSO_LAYERS, PPOUpdateRule, rescaled
from brrrl.algorithms.ppo.types import (
    PPOPolicyOutput,
    PPOTransition,
    RecPPOPolicyInput,
    RecPPOTrajectory,
    rec_example,
)
from brrrl.learning.minibatcher import Sequences
from brrrl.metrics import default_logger
from brrrl.metrics.logger import MultiLogger
from brrrl.metrics.stdout import StdoutLogger
from brrrl.networks.head import ContinuousHead, DiscreteHead, MaskedDiscreteHead, ValueHead
from brrrl.networks.recurrent import Carry, ResetRecurrence
from brrrl.networks.torso import MLPTorso
from brrrl.params.params import ParamStore
from brrrl.types import Params, PRNGKey, VersionedParams
from brrrl.utils.duration import EveryTicks
from brrrl.wrappers.action_mask import ACTION_MASK, OBSERVATION
from brrrl.wrappers.agents import one_agent

HIDDEN_SIZE = 128


@final
class RecPPONetwork(nn.Module):
    """A Torso, a `ResetRecurrence` and the Heads, applied at whatever time width its caller has.

    Acting hands it a length-one time axis and the loss hands it a whole sequence, so there is one
    recurrence rather than an act-time copy that could drift from the learn-time one.
    """

    policy: nn.Module
    hidden_size: int = HIDDEN_SIZE

    @nn.nowrap
    def initialize_carry(self, prefix: tuple[int, ...]) -> Carry:
        """The carry a rollout starts from, reachable without params so a Policy can hold one."""
        return ResetRecurrence(self.hidden_size).initialize_carry(prefix)

    def setup(self):
        self.torso = MLPTorso(TORSO_LAYERS, activation=jax.nn.relu)
        self.recurrence = ResetRecurrence(self.hidden_size)
        self.value = ValueHead()

    def __call__(self, carry: Carry, policy_input: RecPPOPolicyInput) -> tuple[Carry, distrax.Distribution, jax.Array]:
        obs: Any = policy_input.observation
        masked = isinstance(self.policy, MaskedDiscreteHead)
        carry, emb = self.recurrence(carry, self.torso(obs[OBSERVATION] if masked else obs), policy_input.dones)
        return carry, (self.policy(emb, obs[ACTION_MASK]) if masked else self.policy(emb)), self.value(emb)


def make_network(obs_space: gym.Space, act_space: gym.Space, hidden_size: int = HIDDEN_SIZE) -> RecPPONetwork:
    """``act_space`` is one agent's. `brrrl.wrappers.agents.one_agent` of what the env declares."""
    is_masked = isinstance(obs_space, gym.spaces.Dict) and ACTION_MASK in obs_space.spaces
    match act_space:
        case gym.spaces.Discrete() if is_masked:
            return RecPPONetwork(MaskedDiscreteHead(int(act_space.n)), hidden_size)
        case gym.spaces.Discrete():
            return RecPPONetwork(DiscreteHead(int(act_space.n)), hidden_size)
        case gym.spaces.Box(shape=(num_actions,)):
            return RecPPONetwork(ContinuousHead(num_actions), hidden_size)
        case _:
            raise ValueError(f"{act_space} must be discrete or a one-dimensional Box")


def main() -> None:
    """Train on a single box."""
    rollout_len = 128
    num_envs = 8
    num_agents = 1
    num_updates = 5000
    num_epochs = 2
    num_minibatches = 4
    hidden_size = HIDDEN_SIZE
    gamma = 0.99
    learning_rate = 3e-4
    seed = 1
    env_name = "Acrobot-v1"

    key = jax.random.key(seed)
    init_key, policy_key = jax.random.split(key)

    # if the env has an action mask then wrap in brrrl.wrappers.action_mask.ActionMask
    make_env = lambda: rescaled(gym.make(env_name))
    env = make_env()
    stepper = AsyncStepper([make_env] * num_envs, num_agents, wrap=partial(NormalizeReward, gamma=gamma))
    act_space = one_agent(env.action_space, num_agents)
    width = prefix(stepper.num_envs, num_agents)
    example_transition = rec_example(rollout_len, hidden_size, width, env.observation_space, act_space)
    traj_accumulator = TrajectoryAccumulator(example_transition)

    network = make_network(env.observation_space, act_space, hidden_size)

    # A learn-time example, because the scan needs a time axis to trace on
    example_input = RecPPOPolicyInput(
        tree.map(
            lambda leaf: jnp.zeros((1, stepper.num_envs, *jnp.shape(leaf)), jnp.asarray(leaf).dtype),
            env.observation_space.sample(),
        ),
        jnp.zeros((1, *width), bool),
    )
    init_params = lambda weights_key: network.init(weights_key, network.initialize_carry(width), example_input)
    # Annealed to zero across the run
    schedule = optax.linear_schedule(learning_rate, 0.0, num_updates * num_epochs * num_minibatches)
    opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(schedule, eps=ADAM_EPS))

    def learner_forward(
        params: Params, policy_input: RecPPOPolicyInput, per_rollout: Carry
    ) -> tuple[distrax.Distribution, jax.Array]:
        """Learner expects a ``(T, B, ...)`` shaped batch."""
        _, distr, value = network.apply(params, per_rollout, policy_input)  # type: ignore
        return distr, value

    def actor_forward(
        params: Params, policy_input: RecPPOPolicyInput, key: PRNGKey, carry: Carry
    ) -> tuple[PPOPolicyOutput, Carry]:
        """One step, with a length-one time axis for the recurrent network."""
        carry, distr, value = network.apply(params, carry, tree.map(lambda leaf: leaf[None], policy_input))  # type: ignore
        action = distr.sample(seed=key)
        logp = jnp.asarray(distr.log_prob(action))
        return PPOPolicyOutput(action[0], logp[0], value[0]), carry

    updater = PPOUpdateRule(
        learner_forward,
        init_params,
        opt,
        num_envs,
        rollout_len,
        num_epochs,
        num_minibatches,
        RecPPOPolicyInput,
        Sequences(),
        gamma=gamma,
    )
    if not updater.isvalid():
        raise ValueError("Invalid config option selected for PPOUpdateRule")
    update_state = updater.init(init_key)
    update_fn = jax.jit(updater.update)

    param_source = ParamStore(VersionedParams(update_state.params, 0))
    policy = RecPolicy(jax.jit(actor_forward), network.initialize_carry(width), param_source, policy_key)
    roller = Roller(
        stepper, policy, traj_accumulator, RecPPOPolicyInput, PPOTransition, RecPPOTrajectory, rollout_len, seed
    )

    actor_recorder, learner_recorder = default_logger(
        MultiLogger([StdoutLogger()]),  # WandbLogger("brrrl", tags=["PPO", env_name, "recurrent"]),
        every=EveryTicks(20),
    )

    for _ in range(num_updates):
        rollout, act_metrics = roller.rollout()
        actor_recorder.record(act_metrics, num_transitions=roller.num_transitions)
        # One Roller, so the rollout axis a stacked batch of them would bring is width one here.
        stacked = tree.map(lambda leaf: leaf[None], rollout.trajectory)
        update_state, learn_metrics = update_fn(update_state, stacked)
        param_source.publish(update_state.params)
        learner_recorder.record(learn_metrics)

    roller.close()


if __name__ == "__main__":
    main()
