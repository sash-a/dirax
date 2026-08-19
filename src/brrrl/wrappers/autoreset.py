from typing import Any, SupportsFloat, final

import gymnasium as gym

from brrrl.wrappers.agents import over


@final
class Autoreset(gym.Wrapper):
    """`gymnasium.wrappers.Autoreset` for flags that carry the agent axis.

    Checks for episode completion of single or multi-agent envs.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.autoreset = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self.autoreset = False
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, SupportsFloat, Any, Any, dict[str, Any]]:
        if self.autoreset:
            observation, info = self.env.reset()
            self.autoreset = False
            return observation, 0.0, False, False, info
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.autoreset = over(terminated, truncated)
        return observation, reward, terminated, truncated, info
