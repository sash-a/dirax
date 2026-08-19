from typing import Any, final

import gymnasium as gym
import numpy as np
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array

from brrrl.types import StepResult
from brrrl.wrappers.episodes import merge_ep_stats


@final
class Looped(VectorEnv):
    """``gym.Env``s stepped in turn, presented as one vector env: a leading env axis on every field."""

    def __init__(self, envs: list[gym.Env]) -> None:
        self.envs = envs
        self.num_envs = len(envs)
        self.single_observation_space = envs[0].observation_space
        self.single_action_space = envs[0].action_space
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def reset(
        self, *, seed: int | list[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """No episode finishes on a reset, and the episode key is the only one that crosses."""
        resets = [env.reset(seed=s, options=options) for env, s in zip(self.envs, seeded(seed, self.num_envs))]
        return self.stacked_obs([observation for observation, _ in resets]), {}

    def step(self, actions: Any) -> StepResult[Any]:
        """The env axis comes off the actions on the way in and back on to everything on the way out."""
        steps = [env.step(action) for env, action in zip(self.envs, np.asarray(actions), strict=True)]
        observations, rewards, terminations, truncations, infos = zip(*steps, strict=True)
        return (
            self.stacked_obs(list(observations)),
            np.stack(rewards),
            np.stack(terminations),
            np.stack(truncations),
            merge_ep_stats(infos),
        )

    def stacked_obs(self, observations: list[Any]) -> Any:
        """Stacks a list of observations according to the observation space."""
        space = self.single_observation_space
        return concatenate(space, observations, create_empty_array(space, n=self.num_envs))

    def close_extras(self, **kwargs: Any) -> None:
        for env in self.envs:
            env.close()


def seeded(seed: int | list[int] | None, num_envs: int) -> list[int | None]:
    """One seed per env, offset as gymnasium's own vector envs offset theirs so no two envs match."""
    if isinstance(seed, list):
        return list(seed)
    elif isinstance(seed, int):
        return [seed + index for index in range(num_envs)]
    else:  # seed is None
        return [None] * num_envs
