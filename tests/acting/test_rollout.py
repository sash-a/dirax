"""One rollout call with real collaborators, which is where the layout claims are settled.

Real gymnasium vector envs, a real Policy over a real in-memory Store. Only the Transition, the
Trajectory and the record function are test-local — which additionally proves the loop leans on
nothing PPO-shaped.

The counter env makes every slot's provenance assertable by value rather than by shape: an
observation is ``episode + step/10`` and a reward is ``10·episode + step``. Gymnasium is the
oracle for the autoreset stream those values are laid into, deliberately — a scripted tape would
assert the layout against our own assumption about autoreset, and that assumption is precisely
what was wrong twice while this was specified.
"""

from typing import Any, NamedTuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from echo import TrajectoryAccumulator
from gymnasium.vector import SyncVectorEnv
from numpy.typing import NDArray

from brrrl.acting.env_stepper import GymVectorStepper
from brrrl.acting.params import InMemoryParamStore
from brrrl.acting.policy import LocalPolicy
from brrrl.acting.roller import Rollout
from brrrl.types import Array, Params, PRNGKey, VersionedParams

NUM_STEPS = 4
LENGTHS = (2, 3)
NUM_ENVS = len(LENGTHS)

OBSERVATIONS = [[1.0, 1.0], [1.1, 1.1], [1.2, 1.2], [2.0, 1.3], [2.1, 2.0]]
"""What the two counters emit over `NUM_STEPS` steps, the second outliving the first by a step."""

TERMINATED = [[False, False], [False, False], [True, False], [False, True], [False, False]]
REWARDS = [[11.0, 11.0], [12.0, 12.0], [0.0, 13.0], [21.0, 0.0], [0.0, 0.0]]


class Counter(gym.Env):
    """Ends after ``length`` steps. Observation is ``episode + step/10``, reward ``10·episode + step``."""

    def __init__(self, length: int) -> None:
        self.length = length
        self.observation_space = gym.spaces.Box(0.0, np.inf, (1,), np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self.episode = 0
        self.index = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[NDArray, dict]:
        self.episode, self.index = self.episode + 1, 0
        return self.observation(), {}

    def step(self, action: int) -> tuple[NDArray, float, bool, bool, dict]:
        self.index += 1
        return self.observation(), 10.0 * self.episode + self.index, self.index == self.length, False, {}

    def observation(self) -> NDArray:
        return np.array([self.episode + self.index / 10], np.float32)


class Output(NamedTuple):
    action: Array
    value: Array
    log_prob: Array


class Transition(NamedTuple):
    observation: NDArray[np.float32]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    reward: NDArray[np.float32]
    action: Array
    value: Array
    log_prob: Array


class Trajectory(NamedTuple):
    transitions: Transition
    version: NDArray[np.int32]


def act_fn(params: Params, observation: NDArray, key: PRNGKey) -> Output:
    """A pure function of the observation, so the observation → output → slot mapping is proven."""
    value = params["params"]["scale"] * jnp.sum(observation, axis=-1)
    return Output(action=jnp.astype(value, jnp.int32), value=value, log_prob=-value)


def record(
    observation: NDArray[np.float32],
    terminated: NDArray[np.bool_],
    truncated: NDArray[np.bool_],
    reward: NDArray[np.float32],
    output: Output,
) -> Transition:
    return Transition(observation, terminated, truncated, reward, output.action, output.value, output.log_prob)


def example(num_steps: int) -> dict[str, Any]:
    def zeros(dtype: Any, *trailing: int) -> NDArray:
        return np.zeros((num_steps + 1, NUM_ENVS, *trailing), dtype)

    return {
        "transitions": Transition(
            observation=zeros(np.float32, 1),
            terminated=zeros(np.bool_),
            truncated=zeros(np.bool_),
            reward=zeros(np.float32),
            action=zeros(np.int32),
            value=zeros(np.float32),
            log_prob=zeros(np.float32),
        ),
        "version": np.zeros((1,), np.int32),
    }


def build(num_steps: int = NUM_STEPS, version: int = 0) -> Rollout[NDArray, Output, Trajectory]:
    store = InMemoryParamStore(VersionedParams({"params": {"scale": jnp.float32(1.0)}}, version))
    return Rollout(
        stepper=GymVectorStepper(SyncVectorEnv([lambda length=length: Counter(length) for length in LENGTHS])),
        policy=LocalPolicy(act_fn, store, jax.random.key(0)),
        accumulator=TrajectoryAccumulator(example(num_steps)),
        record=record,
        trajectory=Trajectory,
        num_steps=num_steps,
    )


@pytest.fixture
def roller() -> Rollout[NDArray, Output, Trajectory]:
    return build()


def test_a_slot_holds_an_observation_and_the_flags_that_describe_it(
    roller: Rollout[NDArray, Output, Trajectory],
) -> None:
    """The Final obs and the flag saying so are never in different slots, which is what makes the
    unusable slot readable off the flags already in it — a shift of zero rather than of one."""
    trajectory, _ = roller.rollout()

    assert trajectory.transitions.observation[..., 0].tolist() == np.array(OBSERVATIONS, np.float32).tolist()
    assert trajectory.transitions.terminated.tolist() == TERMINATED
    assert not trajectory.transitions.truncated.any()


def test_the_reward_action_and_value_in_a_slot_describe_the_transition_out_of_it(
    roller: Rollout[NDArray, Output, Trajectory],
) -> None:
    """The other half of the alignment: the reward is what the slot's action earned, and the value
    is of the slot's own observation — so truncation bootstrapping reads the value beside the flag."""
    trajectory, _ = roller.rollout()

    transitions = trajectory.transitions
    assert transitions.reward.tolist() == REWARDS
    assert transitions.action.tolist() == [[1, 1], [1, 1], [1, 1], [2, 1], [2, 2]]
    np.testing.assert_allclose(transitions.value, transitions.observation[..., 0])
    np.testing.assert_allclose(transitions.log_prob, -transitions.value)


def test_the_final_obs_slot_is_a_real_observation_with_a_real_value(
    roller: Rollout[NDArray, Output, Trajectory],
) -> None:
    """ADR-0001: only the action and reward beside it are meaningless. Optimising the slot away, or
    filling its observation with the reset that follows, is what this pins down."""
    trajectory, _ = roller.rollout()

    ended, terminal = 2, 0
    assert trajectory.transitions.terminated[ended, terminal]
    assert trajectory.transitions.observation[ended, terminal, 0] == np.float32(1.2)
    assert trajectory.transitions.value[ended, terminal] == np.float32(1.2)
    assert trajectory.transitions.reward[ended, terminal] == 0.0


def test_a_rollout_fills_one_more_slot_than_it_stepped(roller: Rollout[NDArray, Output, Trajectory]) -> None:
    """T+1 in one Timescale, so an Estimator slices a single values array instead of stitching a
    bootstrap onto it. Filled rather than reserved: a loop that stopped at T would leave the last
    slot holding the buffer's zeros, which no observation of this env ever is."""
    trajectory, _ = roller.rollout()

    slots, envs = trajectory.transitions.value.shape
    assert slots == roller.num_transitions // envs + 1
    assert (trajectory.transitions.value != 0).all()


def test_the_trailing_slot_carries_the_flags_the_roller_was_already_holding() -> None:
    """Nothing about the trailing slot is synthesised: its flags may legitimately be set, and here
    the second env's episode ends on the very step the rollout stops at."""
    trajectory, _ = build(num_steps=3).rollout()

    assert trajectory.transitions.terminated[-1].tolist() == [False, True]
    assert trajectory.transitions.observation[-1, ..., 0].tolist() == np.array([2.0, 1.3], np.float32).tolist()
    assert trajectory.transitions.reward[-1].tolist() == [0.0, 0.0]


def test_a_trajectory_carries_exactly_one_version(roller: Rollout[NDArray, Output, Trajectory]) -> None:
    """Its staleness is then a single well-defined number, and rides in the pytree as its own Timescale."""
    trajectory, _ = roller.rollout()

    assert trajectory.version.tolist() == [0]


def test_the_version_is_the_one_the_policy_synced_to_at_the_start_of_the_rollout() -> None:
    """A Trajectory is stamped with the parameters that actually generated it."""
    trajectory, _ = build(version=17).rollout()

    assert trajectory.version.tolist() == [17]


def test_a_roller_reports_how_many_transitions_one_rollout_produces(
    roller: Rollout[NDArray, Output, Trajectory],
) -> None:
    """The Step axis advances by a count the Roller owns. The trailing slot is never stepped, so it
    is not one of them."""
    assert roller.num_transitions == NUM_STEPS * NUM_ENVS


def test_only_completed_episodes_are_reported(roller: Rollout[NDArray, Output, Trajectory]) -> None:
    """One env ran two steps and returned 11+12, the other three and returned 11+12+13."""
    _, metrics = roller.rollout()

    assert np.asarray(metrics["episode_return"]).tolist() == [23.0, 36.0]
    assert np.asarray(metrics["episode_length"]).tolist() == [2, 3]


def test_a_rollout_in_which_nothing_finished_reports_nothing() -> None:
    """Nothing rather than a zero or a nan, which would drag a windowed mean toward it."""
    _, metrics = build(num_steps=1).rollout()

    assert metrics == {}


def test_consecutive_rollouts_are_one_unbroken_stream_of_environment_interaction() -> None:
    """The Roller carries its observation and flags across rollouts and resets exactly once, at the
    start, so no episode is cut short by a Trajectory boundary."""
    roller = build()

    first, _ = roller.rollout()
    trailing = first.transitions.observation[-1].copy()  # the accumulator's buffers are reused below
    second, _ = roller.rollout()

    assert second.transitions.observation[0].tolist() == trailing.tolist()
    assert second.transitions.observation[..., 0].min() >= 2.0
