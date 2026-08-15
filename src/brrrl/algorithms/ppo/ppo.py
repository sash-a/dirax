from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import chex
import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from echo import TrajectoryAccumulator
from jax import tree

from brrrl.acting.env_stepper import GymVectorStepper
from brrrl.acting.policy import LocalPolicy
from brrrl.acting.roller import Roller
from brrrl.algorithms.estimators import generalised_advantages
from brrrl.algorithms.ppo.types import PPOBatch, PPOPolicyOutput, PPOTrajectory, PPOTransition, PPOUpdateState, example
from brrrl.metrics.collector import LocalMetricRecorder, MetricCollector, Origin
from brrrl.metrics.logger import MultiLogger, WindowedProcessor
from brrrl.metrics.stdout import StdoutLogger
from brrrl.metrics.wandb import WandbLogger
from brrrl.networks.head import DiscreteHead, ValueHead
from brrrl.networks.torso import MLPTorso
from brrrl.params.params import InMemoryParamStore
from brrrl.types import Array, ArrayTree, Metrics, Params, PRNGKey, VersionedParams
from brrrl.utils.jax import tree_swapaxes

EPS = 1e-8

type PPOApply = Callable[[Params, ArrayTree], tuple[distrax.Distribution, jax.Array]]
type EpochCarry = tuple[PPOUpdateState, PPOBatch]


def normalise(advantages: Array, mask: Array) -> jax.Array:
    return (advantages - jnp.mean(advantages, where=mask)) / (jnp.sqrt(jnp.var(advantages, where=mask)) + EPS)


def explained_variance(targets: Array, predictions: Array, mask: Array) -> jax.Array:
    """The share of the targets' spread the critic accounts for: 1 is exact, 0 is predicting their mean."""
    spread = jnp.var(targets, where=mask)
    explianed_var = 1 - jnp.var(targets - predictions, where=mask) / spread
    return jnp.where(spread > 0, explianed_var, 0.0)


@dataclass(frozen=True)
class PPOUpdateRule:
    apply_fn: PPOApply
    optimizer: optax.GradientTransformation
    batch_size: int
    rollout_length: int
    num_epochs: int = 2
    num_minibatches: int = 8
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95

    def isvalid(self) -> bool:
        return (
            self.batch_size > 0
            and self.rollout_length > 0
            and self.num_epochs > 0
            and self.num_minibatches > 0
            and (self.batch_size * self.rollout_length) % self.num_minibatches == 0
            and self.clip_eps > 0
            and self.ent_coef >= 0
            and self.vf_coef >= 0
            and 0 <= self.gamma <= 1
            and 0 <= self.gae_lambda <= 1
        )

    @jax.jit(static_argnames=("self",))
    def update(self, state: PPOUpdateState, trajectory: PPOTrajectory) -> tuple[PPOUpdateState, Metrics]:
        """Slots arrive flat: an Actor folds its env axis away, so ``batch_size`` is actors x envs."""
        chex.assert_tree_shape_prefix(trajectory.transitions, (self.batch_size, self.rollout_length + 1))
        advantages = generalised_advantages(
            trajectory.transitions.reward,
            trajectory.transitions.value,
            trajectory.transitions.terminated,
            trajectory.transitions.truncated,
            self.gamma,
            self.gae_lambda,
            axis=1,
        )
        value_targets = advantages + trajectory.transitions.value[:, :-1]
        batch = PPOBatch(tree.map(lambda x: x[:, :-1], trajectory.transitions), advantages, value_targets)
        (state, _), loss_infos = jax.lax.scan(self.epoch, (state, batch), None, self.num_epochs)
        # Explained variance judges the critic that produced these advantages, so it is read once,
        # off the values the rollout acted under, rather than per minibatch off params already moving.
        mask = ~(batch.transitions.terminated | batch.transitions.truncated)
        variance = explained_variance(value_targets, batch.transitions.value, mask)
        return state, tree.map(jnp.mean, loss_infos) | {"explained_variance": variance}

    def shuffle_and_minibatch(self, key: PRNGKey, batch: PPOBatch) -> PPOBatch:
        """One flat pool of steps, permuted and cut into minibatches along a new leading axis."""
        flat_bs = self.batch_size * self.rollout_length
        mb_size = flat_bs // self.num_minibatches
        perm = jax.random.permutation(key, flat_bs)
        return tree.map(
            lambda leaf: (
                leaf.reshape(flat_bs, *leaf.shape[2:])  # flatten T, B
                .take(perm, axis=0)  # shuffle
                .reshape(self.num_minibatches, mb_size, *leaf.shape[2:])  # batch into minibatches
            ),
            batch,
        )

    def epoch(self, carry: EpochCarry, _: None) -> tuple[EpochCarry, Metrics]:
        """The batch rides the carry unchanged: every epoch reshuffles the same steps."""
        state, batch = carry
        key, shuffle_key = jax.random.split(state.key)
        minibatches = self.shuffle_and_minibatch(shuffle_key, batch)
        state, loss_info = jax.lax.scan(self.minibatch, state._replace(key=key), minibatches)
        return (state, batch), loss_info

    def minibatch(self, state: PPOUpdateState, batch: PPOBatch) -> tuple[PPOUpdateState, Metrics]:
        grads, loss_info = jax.grad(self.loss, has_aux=True)(state.params, batch)
        updates, opt_state = self.optimizer.update(grads, state.opt_state, state.params)
        params = cast(Params, optax.apply_updates(state.params, updates))
        return state._replace(params=params, opt_state=opt_state), loss_info

    def loss(self, params: Params, batch: PPOBatch) -> tuple[jax.Array, Metrics]:
        """The clipped surrogate, a clipped value term and an entropy bonus, over the real steps only.

        Written out rather than taken from rlax, whose ``clipped_surrogate_pg_loss`` means over every
        slot it is given and so cannot be told to skip a Final obs.
        """
        transitions = batch.transitions
        mask = ~(transitions.terminated | transitions.truncated)
        distribution, value = self.apply_fn(params, transitions.observation)

        advantages = normalise(batch.advantages, mask)
        log_ratio = distribution.log_prob(transitions.action) - transitions.log_prob
        # Masked inside the exponent rather than after it: masking the result still leaves the
        # gradient multiplying a zero cotangent by an unbounded derivative, which is NaN.
        ratio = jnp.exp(jnp.where(mask, log_ratio, 0.0))
        clipped_ratio = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages), where=mask)

        clipped_value = transitions.value + jnp.clip(value - transitions.value, -self.clip_eps, self.clip_eps)

        squared = jnp.maximum(jnp.square(value - batch.value_targets), jnp.square(clipped_value - batch.value_targets))
        value_loss = 0.5 * jnp.mean(squared, where=mask)
        entropy = jnp.mean(jnp.asarray(distribution.entropy()), where=mask)

        total = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
        metrics = {
            "total_loss": total,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": jnp.mean(ratio - 1 - jnp.log(ratio), where=mask),
            "clip_fraction": jnp.mean(jnp.abs(ratio - 1) > self.clip_eps, where=mask),
        }
        return total, metrics


def main() -> None:
    """Train in one process, one thread."""

    class PPONetwork(nn.Module):
        num_actions: int

        def setup(self):
            self.torso = MLPTorso([64, 64], activation=jax.nn.relu, normalisation=nn.RMSNorm)
            self.policy = DiscreteHead(self.num_actions)
            self.value = ValueHead()

        def __call__(self, obs: jax.Array) -> tuple[distrax.Distribution, jax.Array]:
            emb = self.torso(obs)
            distr = self.policy(emb)
            value = self.value(emb)
            return distr, value

    rollout_len = 128
    num_updates = 5000
    num_envs = 8
    env_name = "Acrobot-v1"

    key = jax.random.key(1)
    weights_key, policy_key, update_key = jax.random.split(key, 3)

    make_env = lambda: gym.make(env_name)
    env = make_env()
    stepper = GymVectorStepper(make_env, num_envs)
    example_transition = example(rollout_len, stepper.num_envs, env.observation_space, env.action_space)
    traj_accumulator = TrajectoryAccumulator(example_transition)

    network = PPONetwork(int(cast(gym.spaces.Discrete, env.action_space).n))
    apply_fn = cast(PPOApply, network.apply)
    params = network.init(weights_key, env.observation_space.sample())
    param_source = InMemoryParamStore(VersionedParams(params, 0))
    opt = optax.adam(3e-4)
    opt_state = opt.init(params)

    def actor_forward(params, observations, key):
        distr, value = apply_fn(params, observations)
        action = distr.sample(seed=key)
        logp = jnp.asarray(distr.log_prob(action))
        return PPOPolicyOutput(action, logp, value)

    def record(obs, term, trunc, reward, policy_output: PPOPolicyOutput):
        return PPOTransition(
            obs, term, trunc, reward, policy_output.action, policy_output.value, policy_output.log_prob
        )

    policy = LocalPolicy(jax.jit(actor_forward), param_source, policy_key)
    roller = Roller(stepper, policy, traj_accumulator, record, PPOTrajectory, rollout_len)
    updater = PPOUpdateRule(apply_fn, opt, num_envs, rollout_len)
    if not updater.isvalid():
        raise ValueError("Invalid config option selected for PPOUpdateRule")
    update_state = PPOUpdateState(params, opt_state, update_key)

    backend = MultiLogger([StdoutLogger(), WandbLogger("brrrl", tags=["PPO", env_name])])
    logger = WindowedProcessor(backend, roller.num_transitions * 20)
    metric_collector = MetricCollector({Origin.ACTOR: logger, Origin.LEARNER: logger})
    actor_recorder = LocalMetricRecorder(metric_collector, Origin.ACTOR)
    learner_recorder = LocalMetricRecorder(metric_collector, Origin.LEARNER)

    for i in range(num_updates):
        trajectory, act_metrics = roller.rollout()
        actor_recorder.record(act_metrics, num_transitions=roller.num_transitions)
        # A Roller rolls out time-major; ``batch_size`` is this worker's envs, so they lead.
        trajectory = trajectory._replace(transitions=tree_swapaxes(trajectory.transitions, 0, 1))
        update_state, learn_metrics = updater.update(update_state, trajectory)
        param_source.publish(update_state.params)
        learner_recorder.record(learn_metrics)


if __name__ == "__main__":
    main()
