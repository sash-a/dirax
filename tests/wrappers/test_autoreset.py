"""Our autoreset against gymnasium's own, which is the oracle ADR-0001's layout was settled against.

It exists at all because gymnasium's asks ``terminated or truncated``, which raises the moment a
flag arrives per agent. Since ours is applied on every path — one agent or many — the first thing it
owes is that the benchmarked single-agent path did not move underneath it. The second is the thing
gymnasium's could not do: read flags that disagree.
"""

from itertools import pairwise
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from brrrl.wrappers.autoreset import Autoreset

STEPS, SEED = 200, 7


def cartpole() -> gym.Env:
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)
    return env


class Team(gym.Env):
    """Ends when every agent has reached the goal, and pays each of them for standing on it.

    Agents arrive on their own schedule, so the flags genuinely disagree for a few steps before the
    episode is over — which is the state gymnasium's autoreset cannot represent at all.
    """

    def __init__(self, arrivals: tuple[int, ...] = (2, 3, 5)) -> None:
        self.arrivals = arrivals
        self.observation_space = gym.spaces.Box(0.0, 1.0, (1,), np.float32)
        self.action_space = gym.spaces.MultiDiscrete([2] * len(arrivals))
        self.elapsed = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict]:
        self.elapsed = 0
        return np.zeros(1, np.float32), {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict]:
        self.elapsed += 1
        arrived = tuple(self.elapsed >= arrival for arrival in self.arrivals)
        return np.full(1, self.elapsed, np.float32), tuple(float(hit) for hit in arrived), arrived, False, {}


def test_a_single_agents_stream_is_gymnasiums_own() -> None:
    """The regression guard on the benchmarked path: ours replaces gymnasium's everywhere, so at one
    agent the two have to be the same wrapper with a different question inside it."""
    ours, theirs = Autoreset(cartpole()), gym.wrappers.Autoreset(cartpole())
    ours.reset()
    theirs.reset()
    rng = np.random.default_rng(0)

    boundaries = 0
    for step in range(STEPS):
        action = int(rng.integers(0, 2))
        mine, theirs_step = ours.step(action), theirs.step(action)
        assert np.array_equal(mine[0], theirs_step[0]), f"observations diverge at step {step}"
        assert mine[1:4] == theirs_step[1:4], f"reward or flags diverge at step {step}"
        boundaries += int(mine[2] or mine[3])

    assert boundaries > 0, "no episode ended, so the comparison above holds vacuously"


def test_the_step_after_an_episode_ends_is_spent_on_the_reset() -> None:
    """ADR-0001's discipline, read off per-agent flags: the reset step pays nothing and flags nothing,
    so the Final obs and the flags describing it are never in different slots."""
    env = Autoreset(Team())
    env.reset()

    steps = [env.step(np.zeros(3, np.int64)) for _ in range(6)]
    after = [step for previous, step in pairwise(steps) if all(previous[2])]

    assert after, "no episode ended, so the assertions below hold vacuously"
    assert all(reward == 0.0 and not terminated and not truncated for _, reward, terminated, truncated, _ in after)


def test_one_agent_reaching_a_terminal_state_does_not_reset_the_world() -> None:
    """Any rather than all would cut the rest of the team's episode short here — at step two, where
    exactly one of three has arrived — and reset a world the other two were still acting in."""
    env = Autoreset(Team())
    env.reset()

    flags = [env.step(np.zeros(3, np.int64))[2] for _ in range(4)]

    assert flags[1] == (True, False, False), "the first agent should have arrived alone"
    assert flags[3] == (True, True, False), "and the episode should still be running"


@pytest.mark.parametrize("field", ["reward", "terminated", "truncated"])
def test_a_reset_step_pays_out_scalars_for_the_axis_above_to_widen(field: str) -> None:
    """It sits under `Agents`, so what it invents on a reset step is broadcast like any other step's
    and it never needs to know how many agents there are."""
    env = Autoreset(Team())
    env.reset()

    steps = [env.step(np.zeros(3, np.int64)) for _ in range(6)]
    reset = next(step for previous, step in pairwise(steps) if all(previous[2]))

    assert np.isscalar(dict(zip(["reward", "terminated", "truncated"], reset[1:4], strict=True))[field])
