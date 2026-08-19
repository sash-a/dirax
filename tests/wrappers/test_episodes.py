"""The one episode-statistics schema, and the recorder that fills it from a reward per agent.

Like the autoreset, this replaces a gymnasium wrapper on every path rather than only the multi-agent
one, so the first thing it owes is that the benchmarked single-agent path did not move. What it adds
is the reduction: a team's episode is one number in the end, and which number is a decision no
default can make for both families of cooperative env.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from brrrl.wrappers.episodes import EPISODE_MASK, EPISODE_STATS, EpisodeStatistics, merge_ep_stats

STEPS, SEED, AGENTS = 200, 7, 3
PAID = (1.0, 0.0, 3.0)
LENGTH = 4


def cartpole() -> gym.Env:
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)
    return env


class Team(gym.Env):
    """Pays each agent a different amount every step, and ends after `LENGTH` of them."""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(0.0, 1.0, (1,), np.float32)
        self.action_space = gym.spaces.MultiDiscrete([2] * AGENTS)
        self.elapsed = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict]:
        self.elapsed = 0
        return np.zeros(1, np.float32), {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict]:
        self.elapsed += 1
        over = self.elapsed >= LENGTH
        return np.zeros(1, np.float32), PAID, (over,) * AGENTS, False, {}


def finished(env: gym.Env, steps: int = LENGTH) -> dict[str, Any]:
    env.reset()
    infos = [env.step(np.zeros(AGENTS, np.int64))[4] for _ in range(steps)]
    return next(info[EPISODE_STATS] for info in infos if EPISODE_STATS in info)


def test_a_single_agents_statistics_are_gymnasiums_own() -> None:
    """The regression guard on the benchmarked path. Ours replaces gymnasium's everywhere, so at one
    agent the two have to report the same episode with the same numbers in it."""
    ours = gym.wrappers.Autoreset(EpisodeStatistics(cartpole()))
    theirs = gym.wrappers.Autoreset(gym.wrappers.RecordEpisodeStatistics(cartpole()))
    ours.reset()
    theirs.reset()
    rng = np.random.default_rng(0)

    reported = 0
    for step in range(STEPS):
        action = int(rng.integers(0, 2))
        mine, theirs_info = ours.step(action)[4], theirs.step(action)[4]
        assert (EPISODE_STATS in mine) == (EPISODE_STATS in theirs_info), f"they disagree at step {step}"
        if EPISODE_STATS in mine:
            reported += 1
            assert mine[EPISODE_STATS]["r"] == theirs_info[EPISODE_STATS]["r"]
            assert mine[EPISODE_STATS]["l"] == theirs_info[EPISODE_STATS]["l"]

    assert reported > 0, "no episode finished, so the comparison above holds vacuously"


def test_a_teams_return_is_the_sum_of_what_its_agents_were_paid() -> None:
    """rware and lbforaging pay each agent its own share, so the team's return is their total and
    the default has to be the sum. A mean here would report a third of what the env paid out."""
    stats = finished(EpisodeStatistics(Team()))

    assert stats["r"] == pytest.approx(sum(PAID) * LENGTH)
    assert stats["l"] == LENGTH


def test_the_reduction_is_the_callers_because_no_default_covers_both_kinds_of_env() -> None:
    """An env broadcasting one shared reward to every agent has that reward multiplied by the agent
    count under a sum. Either default silently misreports the other, so it is a parameter."""
    stats = finished(EpisodeStatistics(Team(), np.mean))

    assert stats["r"] == pytest.approx(sum(PAID) * LENGTH / AGENTS)


def test_an_episode_is_reported_when_every_agent_is_done_and_not_before() -> None:
    """`over` is all, not any. Reporting on the first agent's terminal state would file an episode
    that is still running, and then file the real one a few steps later as well."""
    env = EpisodeStatistics(Team())
    env.reset()

    reported = [EPISODE_STATS in env.step(np.zeros(AGENTS, np.int64))[4] for _ in range(LENGTH)]

    assert reported == [False] * (LENGTH - 1) + [True]


def test_one_envs_statistics_widen_into_the_vector_schema() -> None:
    """A Roller's episode reader knows one schema, so one env owes it the same two keys a vector env
    reports: full-width values, and the mask naming which envs they belong to."""
    widened = merge_ep_stats([{EPISODE_STATS: {"r": 12.0, "l": 4, "t": 0.5}}])

    assert widened[EPISODE_STATS]["r"].tolist() == [12.0]
    assert widened[EPISODE_MASK].tolist() == [True]


def test_an_info_that_finished_no_episode_widens_into_nothing() -> None:
    """Inventing a zero for a step that finished nothing would report an episode that never happened
    and drag every windowed mean toward it."""
    assert merge_ep_stats([{"other": 1}]) == {}


def test_a_loop_over_envs_reports_the_same_schema() -> None:
    """One env is the same call at width one. Which envs finished is the mask's to say, so the values
    in the slots that did not are never read and never have to mean anything."""
    aggregated = merge_ep_stats([{}, {EPISODE_STATS: {"r": 3.0, "l": 2, "t": 0.1}}, {}])

    assert aggregated[EPISODE_MASK].tolist() == [False, True, False]
    assert aggregated[EPISODE_STATS]["r"].tolist() == [0.0, 3.0, 0.0]


def test_a_loop_in_which_nothing_finished_reports_no_keys_at_all() -> None:
    assert merge_ep_stats([{}, {}]) == {}
