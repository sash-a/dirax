import time
from collections.abc import Callable, Sequence
from typing import Any, SupportsFloat, final

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from brrrl.wrappers.agents import over

EPISODE_STATS = "episode"
EPISODE_MASK = "_episode"

# How an episode's per-agent returns become the one number a run is judged on.
type Reduction = Callable[[NDArray[np.floating]], np.floating]


@final
class EpisodeStatistics(gym.Wrapper):
    """`gymnasium.wrappers.RecordEpisodeStatistics` for a reward that carries the agent axis.

    The same three keys under the same key, because a Roller reads one schema whatever built the
    environment. Like `Autoreset` it asks `over` rather than ``terminated or truncated``, and like
    `Autoreset` to keep the API consistent.

    ``reduce`` is what makes an episode's per-agent returns the one number a run is judged on. It
    allows for configuration of sum or mean of returns depending on the use case.
    """

    def __init__(self, env: gym.Env, reduce: Reduction = np.sum) -> None:
        super().__init__(env)
        self.reduce = reduce
        self.restart()

    def restart(self) -> None:
        self.returns = np.zeros(())
        self.length = 0
        self.started = time.perf_counter()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self.restart()
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, SupportsFloat, Any, Any, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.returns = self.returns + np.asarray(reward, np.float32)
        self.length += 1
        if over(terminated, truncated):
            info = info | {EPISODE_STATS: self.statistics()}
        return observation, reward, terminated, truncated, info

    def statistics(self) -> dict[str, float | int]:
        return {
            "r": float(self.reduce(self.returns)),
            "l": self.length,
            "t": round(time.perf_counter() - self.started, 6),
        }


def merge_ep_stats(infos: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """One env-per-info loop's episode statistics in the schema a vector env reports them in."""
    finished = np.array([EPISODE_STATS in info for info in infos], np.bool_)
    if not finished.any():
        return {}
    keys = next(info[EPISODE_STATS] for info in infos if EPISODE_STATS in info)
    stats = {key: np.array([info.get(EPISODE_STATS, {}).get(key, 0) for info in infos], np.float32) for key in keys}
    return {EPISODE_STATS: stats, EPISODE_MASK: finished}
