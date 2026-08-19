"""One rollout call with real collaborators, which is where the layout claims are settled.

Real environments, a real Policy over a real in-memory Store. Only the Transition, the Trajectory
and the record function are test-local — which additionally proves the loop leans on nothing
PPO-shaped.

The counter envs make every slot's provenance assertable by value rather than by shape: an
observation is ``episode + step/10`` and agent ``i``'s reward is ``10·episode + step + i``.
Gymnasium's own vector env is the oracle for the autoreset stream those values are laid into,
deliberately — a scripted tape would assert the layout against our own assumption about autoreset,
and that assumption is precisely what was wrong twice while this was specified. Nothing in ``src``
runs on that vector env any more, so the oracle lives here: what ships is held to it rather than
built on it.
"""

from contextlib import closing
from dataclasses import dataclass
from typing import Any, NamedTuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from echo import TrajectoryAccumulator
from gymnasium.vector import AutoresetMode
from numpy.typing import ArrayLike, NDArray

from brrrl.acting.env_stepper import AsyncStepper, EnvStepper, LoopedStepper, prefix
from brrrl.acting.policy import LocalPolicy
from brrrl.acting.roller import Roller
from brrrl.params.params import InMemoryParamStore, ParamSource
from brrrl.types import Array, Params, PRNGKey, VersionedParams

NUM_STEPS = 4
LENGTHS = (2, 3)
NUM_ENVS = len(LENGTHS)
AGENTS = 3

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


class Counters(gym.Env):
    """`Counter` for a team: agent ``i`` is paid ``10·episode + step + i``, and ends ``i·stagger`` late.

    Every agent's slot is distinguishable from every other's by value, so an agent-axis assertion is
    about which number landed where rather than only about a shape. ``stagger`` is what makes the
    agents disagree about being finished, which is the case a single flag per env cannot represent.
    """

    def __init__(self, length: int, num_agents: int = AGENTS, stagger: int = 0) -> None:
        self.length, self.num_agents, self.stagger = length, num_agents, stagger
        self.observation_space = gym.spaces.Tuple([gym.spaces.Box(0.0, np.inf, (1,), np.float32)] * num_agents)
        self.action_space = gym.spaces.MultiDiscrete([4] * num_agents)
        self.episode = 0
        self.index = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict]:
        self.episode, self.index = self.episode + 1, 0
        return self.observation(), {}

    def step(self, action: Any) -> tuple[Any, tuple[float, ...], tuple[bool, ...], bool, dict]:  # ty: ignore[invalid-method-override]
        self.index += 1
        reward = tuple(10.0 * self.episode + self.index + agent for agent in range(self.num_agents))
        ends = tuple(self.index >= self.length + agent * self.stagger for agent in range(self.num_agents))
        return self.observation(), reward, ends, False, {}

    def observation(self) -> tuple[NDArray, ...]:
        return tuple(np.array([self.episode + self.index / 10], np.float32) for _ in range(self.num_agents))


class Oracle:
    """gymnasium's own vector env, presented as an `EnvStepper`. The stepper family's reference.

    ADR-0001's layout was settled against this and the family used to be built on it. It cannot
    carry a team — it assigns each env's reward into a scalar slot of an ``(E,)`` buffer — so what
    ships stepped away from it, and this is what holds the single-agent stream it left behind.
    """

    num_agents = 1

    def __init__(self, make_env, num_envs: int) -> None:
        self.env = gym.vector.SyncVectorEnv(
            [lambda: gym.wrappers.RecordEpisodeStatistics(make_env())] * num_envs,
            autoreset_mode=AutoresetMode.NEXT_STEP,
        )

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def reset(self) -> tuple[Any, dict[str, Any]]:
        return self.env.reset()

    def step(self, action: Any) -> tuple[Any, NDArray, NDArray, NDArray, dict[str, Any]]:
        """Nothing is reshaped either way: an env with a single agent has no agent axis, so what
        gymnasium's vector env reports is already the shape a Roller reads."""
        return self.env.step(np.asarray(action))

    def close(self) -> None:
        self.env.close()


class Output(NamedTuple):
    action: Array
    value: Array
    log_prob: Array


class Transition(NamedTuple):
    observation: Any
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    reward: NDArray[np.float32]
    action: Array
    value: Array
    log_prob: Array


class Trajectory(NamedTuple):
    transitions: Transition
    version: NDArray[np.float32]


def act_fn(params: Params, observation: Any, key: PRNGKey) -> Output:
    """A pure function of the observation, so the observation → output → slot mapping is proven.

    It says nothing about agents, which is the point. A team's env hands it one observation per agent
    and it returns a value per agent; a single agent's env hands it one and it returns one. Stacking
    the tuple is unpacking the observation, which ADR-0008 makes the algorithm's job — and there is
    no axis to invent on either side of it.
    """
    features = jnp.stack(observation, axis=-2) if isinstance(observation, tuple) else jnp.asarray(observation)
    value = params["params"]["scale"] * jnp.sum(features, axis=-1)
    return Output(action=jnp.astype(value, jnp.int32), value=value, log_prob=-value)


def record(
    observation: Any,
    terminated: NDArray[np.bool_],
    truncated: NDArray[np.bool_],
    reward: NDArray[np.float32],
    output: Output,
) -> Transition:
    return Transition(observation, terminated, truncated, reward, output.action, output.value, output.log_prob)


def example(num_steps: int, num_envs: int, num_agents: int = 1, leaves: int = 1) -> dict[str, Any]:
    """Every field leads with what the stepper reports; the observation keeps its env's own shape."""
    width = prefix(num_envs, num_agents)

    def zeros(dtype: Any, *trailing: int) -> NDArray:
        return np.zeros((num_steps + 1, *width, *trailing), dtype)

    def observed() -> NDArray:
        return np.zeros((num_steps + 1, num_envs, 1), np.float32)

    observation = observed() if leaves == 1 else tuple(observed() for _ in range(leaves))
    return {
        "transitions": Transition(
            observation=observation,
            terminated=zeros(np.bool_),
            truncated=zeros(np.bool_),
            reward=zeros(np.float32),
            action=zeros(np.int32),
            value=zeros(np.float32),
            log_prob=zeros(np.float32),
        ),
        "version": np.zeros((1,), np.float32),
    }


def counters() -> LoopedStepper[NDArray, ArrayLike]:
    """The stepper calls each maker once, in order, so the lengths come off an iterator."""
    lengths = iter(LENGTHS)
    return LoopedStepper([lambda: Counter(next(lengths))] * len(LENGTHS))


def teams(stagger: int = 0) -> LoopedStepper[Any, ArrayLike]:
    lengths = iter(LENGTHS)
    return LoopedStepper([lambda: Counters(next(lengths), stagger=stagger)] * len(LENGTHS), AGENTS)


@dataclass
class Stepping:
    """A Source that has moved on by the time it is next read, so one rollout straddles publishes.

    Its first read is the Policy's own, on construction; the reads after that are its acts.
    """

    versions: list[int]
    reads: int = 0

    def get(self, block: bool) -> VersionedParams:
        version = self.versions[min(self.reads, len(self.versions) - 1)]
        self.reads += 1
        return VersionedParams({"params": {"scale": jnp.float32(1.0)}}, version)


def build(
    num_steps: int = NUM_STEPS,
    version: int = 0,
    stepper: EnvStepper[Any, ArrayLike] | None = None,
    source: ParamSource | None = None,
    leaves: int = 1,
) -> Roller[Any, Output, Trajectory]:
    stepper = counters() if stepper is None else stepper
    store = InMemoryParamStore(VersionedParams({"params": {"scale": jnp.float32(1.0)}}, version))
    return Roller(
        stepper=stepper,
        policy=LocalPolicy(act_fn, store if source is None else source, jax.random.key(0)),
        accumulator=TrajectoryAccumulator(example(num_steps, stepper.num_envs, stepper.num_agents, leaves)),
        record=record,
        trajectory=Trajectory,
        num_steps=num_steps,
    )


@pytest.fixture
def roller() -> Roller[Any, Output, Trajectory]:
    return build()


def test_a_slot_holds_an_observation_and_the_flags_that_describe_it(
    roller: Roller[Any, Output, Trajectory],
) -> None:
    """The Final obs and the flag saying so are never in different slots, which is what makes the
    unusable slot readable off the flags already in it — a shift of zero rather than of one."""
    trajectory, _ = roller.rollout()

    assert trajectory.transitions.observation[..., 0].tolist() == np.array(OBSERVATIONS, np.float32).tolist()
    assert trajectory.transitions.terminated.tolist() == TERMINATED
    assert not trajectory.transitions.truncated.any()


def test_the_reward_action_and_value_in_a_slot_describe_the_transition_out_of_it(
    roller: Roller[Any, Output, Trajectory],
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
    roller: Roller[Any, Output, Trajectory],
) -> None:
    """ADR-0001: only the action and reward beside it are meaningless. Optimising the slot away, or
    filling its observation with the reset that follows, is what this pins down."""
    trajectory, _ = roller.rollout()

    ended, terminal = 2, 0
    assert trajectory.transitions.terminated[ended, terminal]
    assert trajectory.transitions.observation[ended, terminal, 0] == np.float32(1.2)
    assert trajectory.transitions.value[ended, terminal] == np.float32(1.2)
    assert trajectory.transitions.reward[ended, terminal] == 0.0


def test_a_rollout_fills_one_more_slot_than_it_stepped(roller: Roller[Any, Output, Trajectory]) -> None:
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


def test_a_trajectory_carries_the_version_its_acts_ran_under(roller: Roller[Any, Output, Trajectory]) -> None:
    """It rides in the pytree as its own Timescale, so a Trainer can log staleness across a batch
    rather than as a second series that cannot be joined to the first."""
    trajectory, _ = roller.rollout()

    assert trajectory.version.tolist() == [0.0]


def test_a_trajectory_that_spanned_no_publish_carries_that_one_version() -> None:
    """A Trajectory is stamped with the parameters that actually generated it."""
    trajectory, _ = build(version=17).rollout()

    assert trajectory.version.tolist() == [17.0]


def test_a_trajectory_that_spanned_a_publish_carries_the_mean_of_the_versions_it_acted_under() -> None:
    """A Policy pulls at its own rate, so a rollout can straddle a publish. The mean is what makes
    that a number a Trainer can subtract from its own Version and read as staleness."""
    acted_under = [4, 4, 6, 6, 6]

    trajectory, _ = build(source=Stepping([0, *acted_under])).rollout()

    assert trajectory.version.tolist() == [pytest.approx(5.2)]


def test_a_roller_reports_how_many_transitions_one_rollout_produces(
    roller: Roller[Any, Output, Trajectory],
) -> None:
    """The Step axis advances by a count the Roller owns. The trailing slot is never stepped, so it
    is not one of them."""
    assert roller.num_transitions == NUM_STEPS * NUM_ENVS


def test_a_team_is_not_a_multiplier_on_the_transition_count() -> None:
    """A Transition is an environment step, however many agents took it. Multiplying the agent count
    in would put every multi-agent curve a factor of ``A`` along the Step axis from the published one
    it is being read against, which is not a thing a reader can spot."""
    with closing(teams()) as stepper:
        assert build(stepper=stepper, leaves=AGENTS).num_transitions == NUM_STEPS * NUM_ENVS


def test_only_completed_episodes_are_reported(roller: Roller[Any, Output, Trajectory]) -> None:
    """One env ran two steps and returned 11+12, the other three and returned 11+12+13."""
    _, metrics = roller.rollout()

    assert np.asarray(metrics["episode_return"]).tolist() == [23.0, 36.0]
    assert np.asarray(metrics["episode_length"]).tolist() == [2, 3]


def test_a_rollout_in_which_nothing_finished_reports_nothing() -> None:
    """Nothing rather than a zero or a nan, which would drag a windowed mean toward it."""
    _, metrics = build(num_steps=1).rollout()

    assert metrics == {}


@pytest.mark.parametrize(
    "stepper",
    [
        pytest.param(lambda: LoopedStepper([lambda: Counter(LENGTHS[0])]), id="loop"),
        pytest.param(lambda: AsyncStepper([lambda: Counter(LENGTHS[0])]), id="processes"),
    ],
)
def test_every_stepper_produces_the_trajectory_gymnasiums_own_vector_env_does(stepper) -> None:
    """The family implements next-step autoreset itself rather than asserting one, so gymnasium's
    own is the oracle it is held to — the same oracle the layout was settled against. A hand-rolled
    reset that fired a step early would diverge here, and so would a third copy of it drifting."""
    with closing(stepper()) as built, closing(Oracle(lambda: Counter(LENGTHS[0]), 1)) as oracle:
        ours, _ = build(stepper=built).rollout()
        theirs, _ = build(stepper=oracle).rollout()

        for one, other in zip(ours.transitions, theirs.transitions, strict=True):
            np.testing.assert_array_equal(one, other)


@pytest.mark.parametrize(
    "stepper",
    [
        pytest.param(lambda: LoopedStepper([lambda: Counter(LENGTHS[0])]), id="loop"),
        pytest.param(lambda: AsyncStepper([lambda: Counter(LENGTHS[0])]), id="processes"),
    ],
)
def test_every_stepper_reports_the_episodes_gymnasiums_own_vector_env_does(stepper) -> None:
    """Our episode statistics replace gymnasium's on every path now, so what they report has to be
    what gymnasium's reported before they did."""
    with closing(stepper()) as built, closing(Oracle(lambda: Counter(LENGTHS[0]), 1)) as oracle:
        _, ours = build(stepper=built).rollout()
        _, theirs = build(stepper=oracle).rollout()

        assert ours.keys() == theirs.keys()
        assert np.asarray(ours["episode_return"]).tolist() == np.asarray(theirs["episode_return"]).tolist()
        assert np.asarray(ours["episode_length"]).tolist() == np.asarray(theirs["episode_length"]).tolist()


def test_consecutive_rollouts_are_one_unbroken_stream_of_environment_interaction() -> None:
    """The Roller carries its observation and flags across rollouts and resets exactly once, at the
    start, so no episode is cut short by a Trajectory boundary."""
    roller = build()

    first, _ = roller.rollout()
    trailing = first.transitions.observation[-1].copy()  # the accumulator's buffers are reused below
    second, _ = roller.rollout()

    assert second.transitions.observation[0].tolist() == trailing.tolist()
    assert second.transitions.observation[..., 0].min() >= 2.0


def test_every_field_but_the_observation_carries_the_agent_axis() -> None:
    """``(T+1, E, A)`` on everything an Estimator or a mask reads. The observation keeps the shape
    the env gave it, because no stepper touches one and only the Assembly reads it."""
    with closing(teams()) as stepper:
        trajectory, _ = build(stepper=stepper, leaves=AGENTS).rollout()

        transitions = trajectory.transitions
        width = (NUM_STEPS + 1, NUM_ENVS, AGENTS)
        assert transitions.reward.shape == transitions.value.shape == width
        assert transitions.terminated.shape == transitions.truncated.shape == width
        assert [leaf.shape for leaf in transitions.observation] == [(NUM_STEPS + 1, NUM_ENVS, 1)] * AGENTS


def test_each_agents_reward_lands_in_its_own_slot() -> None:
    """Agent ``i`` is paid ``i`` more than agent zero every step, so a reward array that had been
    broadcast from one agent's, or transposed, reads differently here."""
    with closing(teams()) as stepper:
        trajectory, _ = build(stepper=stepper, leaves=AGENTS).rollout()

        first = trajectory.transitions.reward[0]
        assert first.tolist() == [[11.0, 12.0, 13.0]] * NUM_ENVS


def test_agents_that_terminate_on_different_steps_keep_the_disagreement() -> None:
    """SMAX's case. A flag collapsed to one bool per env and broadcast back would lose this, and a
    dead agent's slots would go unmasked for as long as the rest of the team kept playing."""
    with closing(teams(stagger=1)) as stepper:
        trajectory, _ = build(stepper=stepper, leaves=AGENTS).rollout()

        ends = trajectory.transitions.terminated[:, 0].tolist()
        assert [True, False, False] in ends, f"the first agent should finish alone, got {ends}"


def test_a_teams_episode_is_over_when_every_agent_is_done() -> None:
    """All, not any: the first env's agents finish on steps 2, 3 and 4, and the episode is the
    longest of them rather than the shortest."""
    with closing(teams(stagger=1)) as stepper:
        _, metrics = build(stepper=stepper, num_steps=6, leaves=AGENTS).rollout()

        assert np.asarray(metrics["episode_length"]).tolist()[0] == LENGTHS[0] + AGENTS - 1
