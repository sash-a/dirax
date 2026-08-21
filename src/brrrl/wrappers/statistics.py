import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any, SupportsFloat, final

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from brrrl.types import Metrics
from brrrl.wrappers.agents import over

EPISODE_STATS = "episode"
ENV_STATS = "env_stats"
EPISODE_MASK = "_episode"
EPISODE_METRICS = {"r": "episode_return", "l": "episode_length", "t": "episode_time"}

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


def episode_stats(infos: Iterable[Mapping[str, Any]]) -> Metrics:
    """Every episode that finished over a rollout, read out of the statistics schema.

    A rollout in which none finished reports no keys at all rather than a zero or a nan.
    """
    finished = [(info[EPISODE_STATS], info[EPISODE_MASK]) for info in infos if EPISODE_STATS in info]
    if not finished:
        return {}
    return {
        name: np.concatenate([stats[key][mask] for stats, mask in finished]) for key, name in EPISODE_METRICS.items()
    }


def env_stats(infos: Iterable[Mapping[str, Any]]) -> Metrics:
    """Whatever an env reported for itself over a rollout, one array per name, read from `ENV_STATS`.

    Every metric arrives beside the ``_name`` mask saying which envs reported it, so the slots a
    vector env filled for the envs that did not are dropped rather than meaned in as zeros. A rollout
    in which nothing reported has no keys at all, for the same reason `episode_stats` has none.
    """
    reported = defaultdict(list)
    for info in infos:
        stats = info.get(ENV_STATS, {})
        for name in stats:
            if not name.startswith("_"):
                reported[name].append(np.asarray(stats[name])[stats[f"_{name}"]])
    return {name: numeric(np.concatenate(values)) for name, values in reported.items()}


def numeric(values: NDArray) -> NDArray:
    """``values`` as the numbers they were reported as.

    A vector env files a type it does not recognise into an object array, so an env reporting a list
    where another reports an array of the same numbers arrives differently for no reason a caller
    could see. Reading one back through ``tolist`` is what makes the two agree.
    """
    return np.array(values.tolist()) if values.dtype == object else values
