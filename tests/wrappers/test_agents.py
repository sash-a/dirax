"""The agent axis: one broadcast per field, and the cases it has to cover with no branch.

An env with a team acting in it gets the axis on every field; an env with a single agent gets no
axis at all, and is not wrapped. What varies within a team is per *field*: rware and lbforaging pay
each agent its own reward and then end for everyone at once, so an env picks a shape per field
rather than one for all of them, and the mixed case has to be free rather than a third code path.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from brrrl.wrappers.agents import Agents, one_agent, over

AGENTS = 3
OBS = 2


class Fixed(gym.Env):
    """Steps to exactly what it was built with, so a test names the shape an env reports in.

    It records what arrived as an action, which is the only thing in the chain that goes the other
    way and the only thing this wrapper does to one.
    """

    def __init__(self, reward: Any, terminated: Any = False, truncated: Any = False, num_agents: int = AGENTS) -> None:
        self.observation_space = gym.spaces.Box(0.0, 1.0, (OBS,), np.float32)
        self.action_space = gym.spaces.MultiDiscrete([2] * num_agents)
        self.paid, self.ended, self.cut = reward, terminated, truncated
        self.received: list[Any] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict]:
        return np.zeros(OBS, np.float32), {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict]:
        self.received.append(action)
        return np.zeros(OBS, np.float32), self.paid, self.ended, self.cut, {}


def stepped(env: gym.Env, num_agents: int = AGENTS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrapped = Agents(env, num_agents)
    wrapped.reset()
    _, reward, terminated, truncated, _ = wrapped.step(np.zeros(num_agents, np.int64))
    return reward, terminated, truncated


def test_a_reward_the_team_shares_is_broadcast_across_it() -> None:
    """The cooperative case where one number is paid to everyone: every agent is credited with it,
    rather than one agent being credited and the rest with nothing."""
    reward, _, _ = stepped(Fixed(2.0))

    assert reward.tolist() == [2.0] * AGENTS


def test_a_reward_already_paid_per_agent_is_left_alone() -> None:
    """rware and lbforaging's shape, and the broadcast is a no-op on it — which is the whole reason
    there is no branch between the cases."""
    reward, _, _ = stepped(Fixed((1.0, 0.0, 3.0)))

    assert reward.tolist() == [1.0, 0.0, 3.0]


def test_every_field_answers_for_itself() -> None:
    """The mixed env, which is the common one: paid per agent, and over for everyone at once. It
    costs no more code than a pure case because the broadcast is applied per field."""
    reward, terminated, truncated = stepped(Fixed((1.0, 0.0, 3.0), terminated=True))

    assert reward.tolist() == [1.0, 0.0, 3.0]
    assert terminated.tolist() == [True] * AGENTS
    assert truncated.tolist() == [False] * AGENTS


def test_agents_that_disagree_about_terminating_keep_the_disagreement() -> None:
    """SMAX's case, and PettingZoo's parallel API. Collapsing the flags to one bool per env and
    broadcasting it back would discard this silently, and a dead agent's slots would go unmasked."""
    _, terminated, _ = stepped(Fixed(1.0, terminated=(True, False, True)))

    assert terminated.tolist() == [True, False, True]


def test_a_flag_of_the_wrong_width_is_refused() -> None:
    """An env reporting a different number of agents than it was declared with is a mistake to hear
    about, not one to stretch to fit."""
    with pytest.raises(ValueError, match="broadcast"):
        stepped(Fixed(1.0, terminated=(True, False)))


def test_the_reward_arrives_as_a_float_whatever_the_env_paid_in() -> None:
    """A Trajectory's reward slot is float32, so a python float or an int reward would be cast on
    every write otherwise — silently, and on the hot path."""
    reward, terminated, _ = stepped(Fixed((1, 0, 3), terminated=1))

    assert reward.dtype == np.float32
    assert terminated.dtype == np.bool_


@pytest.mark.parametrize("num_agents", [0, 1], ids=["none", "one"])
def test_an_env_without_a_team_is_refused(num_agents: int) -> None:
    """There is no axis to widen to below two agents. A single agent's fields carry no agent axis at
    all, and giving one a width-one axis would only oblige every network written for that env to
    produce the axis back — which is what this whole wrapper exists to spare them (ADR-0012)."""
    with pytest.raises(ValueError, match="fewer than two agents"):
        Agents(Fixed(0.0, num_agents=max(num_agents, 1)), num_agents)


def test_a_teams_actions_go_through_untouched() -> None:
    """The array is the env's problem: one that cannot take an array of actions brings its own
    wrapper, because the action goes into an env that is not ours."""
    env = Fixed(0.0)
    wrapped = Agents(env, AGENTS)

    wrapped.step(np.array([1, 2, 0]))

    assert np.asarray(env.received[0]).tolist() == [1, 2, 0]


def test_the_observation_is_not_touched() -> None:
    """It is read by exactly one thing, and ADR-0008 makes unpacking it the algorithm's job — so a
    team's observation reaches a network at whatever shape its env gave it."""
    env = Fixed(0.0)
    wrapped = Agents(env, AGENTS)

    observation, *_ = wrapped.step(np.zeros(AGENTS, np.int64))

    assert observation.shape == (OBS,)
    assert wrapped.observation_space == env.observation_space


@pytest.mark.parametrize(
    ("terminated", "truncated", "finished"),
    [
        ([False, False, False], [False, False, False], False),
        ([True, False, False], [False, False, False], False),
        ([True, True, False], [False, False, False], False),
        ([True, True, True], [False, False, False], True),
        ([True, False, True], [False, True, False], True),
        (False, False, False),
        (True, False, True),
    ],
    ids=["none", "one", "most", "all", "mixed kinds", "single agent running", "single agent done"],
)
def test_over_is_every_agent_and_not_any(terminated: Any, truncated: Any, finished: bool) -> None:
    """One agent reaching a terminal state is not the environment reaching one. Any would end the
    episode the first time an agent died, cutting the rest of the team's episode short and resetting
    a world they were still acting in.

    It still answers for a single agent's bare bools, because the wrappers that ask it are on every
    env whether or not `Agents` is.
    """
    assert over(terminated, truncated) is finished


@pytest.mark.parametrize(
    ("space", "num_agents", "expected"),
    [
        (gym.spaces.MultiDiscrete([3] * AGENTS), AGENTS, gym.spaces.Discrete(3)),
        (gym.spaces.Tuple([gym.spaces.Discrete(3)] * AGENTS), AGENTS, gym.spaces.Discrete(3)),
        (gym.spaces.Discrete(3), 1, gym.spaces.Discrete(3)),
        (gym.spaces.Box(-1.0, 1.0, (OBS,), np.float32), 1, gym.spaces.Box(-1.0, 1.0, (OBS,), np.float32)),
    ],
    ids=["multi discrete", "tuple", "one discrete", "one box"],
)
def test_one_agents_action_space_is_what_a_head_is_built_from(
    space: gym.Space, num_agents: int, expected: gym.Space
) -> None:
    """A Head emits one agent's action and the axis is the buffer's, so the space a network is built
    from names one agent however the env named the team's."""
    assert one_agent(space, num_agents) == expected


def test_one_agents_continuous_action_space_keeps_its_own_axis() -> None:
    """A continuous action has a width of its own, and only the leading axis is the team's."""
    space = one_agent(gym.spaces.Box(-1.0, 1.0, (AGENTS, OBS), np.float32), AGENTS)

    assert space.shape == (OBS,)


def test_an_action_space_that_does_not_name_the_team_is_refused() -> None:
    """Guessing here would build a Head for the wrong number of actions, which trains."""
    with pytest.raises(TypeError, match=f"{AGENTS} agents"):
        one_agent(gym.spaces.MultiDiscrete([3, 3]), AGENTS)
