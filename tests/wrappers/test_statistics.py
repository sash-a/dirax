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

from brrrl.wrappers.statistics import ENV_STATS, EPISODE_STATS, EpisodeStatistics, env_stats

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


def reported(*envs: float | None) -> dict[str, Any]:
    """One step's info for a metric that only some envs reported, in the vector schema.

    A vector env fills a slot for every env and names the ones that meant it in the ``_name`` mask,
    so the zeros below are what an env that reported nothing leaves behind.
    """
    mask = [value is not None for value in envs]
    return {ENV_STATS: {"shaped": np.array([value or 0.0 for value in envs]), "_shaped": np.array(mask)}}


def test_a_metric_only_some_envs_reported_drops_the_slots_they_left() -> None:
    """The zero a vector env fills an unreported slot with is filler, not data. Meaning it in would
    drag the metric toward zero by however many envs were quiet, which no reader could spot."""
    assert np.asarray(env_stats([reported(100.0, None)])["shaped"]).tolist() == [100.0]


def test_the_masks_are_read_rather_than_reported() -> None:
    """They arrive beside every metric a vector env aggregates. Logged as metrics themselves they
    would double the series count, and each would read as a clean number rather than an obvious
    mistake — the mean of a bool mask is a fraction between zero and one."""
    assert list(env_stats([reported(1.0, 2.0)])) == ["shaped"]


def test_every_step_of_a_rollout_lands_in_one_array() -> None:
    """What a Roller hands the reader is the whole rollout, so a metric reported at several steps is
    one series rather than one per step."""
    assert np.asarray(env_stats([reported(1.0, 2.0), reported(None, 3.0)])["shaped"]).tolist() == [1.0, 2.0, 3.0]


def filed(*values: list[int]) -> np.ndarray:
    """One env's list-valued metric per slot, as a vector env files a type it does not recognise."""
    array = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return array


def test_a_metric_reported_as_a_list_reads_as_the_numbers_it_is() -> None:
    """A vector env files a type it does not recognise into an object array, so two envs reporting the
    same numbers — one as a list, one as an array — would arrive differently. The accumulator squares
    what it folds, which an object array cannot do, so the run dies at the first record rather than
    where the list was written."""
    listed = {ENV_STATS: {"pair": filed([1, 2], [3, 4]), "_pair": np.array([True, True])}}
    arrayed = {ENV_STATS: {"pair": np.array([[1, 2], [3, 4]]), "_pair": np.array([True, True])}}

    assert np.asarray(env_stats([listed])["pair"]).dtype != object
    assert np.asarray(env_stats([listed])["pair"]).tolist() == np.asarray(env_stats([arrayed])["pair"]).tolist()


def test_a_rollout_that_reported_nothing_has_no_keys_at_all() -> None:
    """The same silence `episode_stats` keeps, and for the same reason: a zero here is an env metric
    that never happened."""
    assert env_stats([{}, {EPISODE_STATS: {"r": 1.0}}]) == {}
