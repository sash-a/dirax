from typing import Any, final

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array

from brrrl.types import StepResult

EPISODE_STATS = "episode"
EPISODE_MASK = "_episode"
"""What a vector env reports its finished episodes under: full-width statistics, and the mask naming
which envs they belong to."""


def full_width(info: dict[str, Any]) -> dict[str, Any]:
    """One env's episode statistics in the schema a vector env reports them in.

    The single-env statistics wrapper reports scalars and no mask, so this is what one env owes a
    Roller for its episode reader to stay the only place that knows the schema.
    """
    if EPISODE_STATS not in info:
        return info
    stats = info[EPISODE_STATS]
    return info | {
        EPISODE_STATS: {key: np.array([value]) for key, value in stats.items()},
        EPISODE_MASK: np.ones(1, np.bool_),
    }


@final
class Widened(gym.Wrapper):
    """One `gym.Env` stepped through the vector API: a leading env axis of one on every field.

    It goes on outermost, and gymnasium's own wrappers all go under it: every field it returns is an
    array, where a `gym.Env` returns a scalar reward and two bare flags. That is the whole of what a
    single env has to answer for a Roller to read it and a vectorised one identically, so the
    difference between the two stops here rather than reaching an algorithm.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = batch_space(env.observation_space, 1)
        self.action_space = batch_space(env.action_space, 1)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        return self.batched(observation), full_width(info)

    def step(self, action: Any) -> StepResult[Any]:  # ty: ignore[invalid-method-override]
        """The env axis comes off the action on the way in and back on to everything on the way out."""
        observation, reward, terminated, truncated, info = self.env.step(np.asarray(action)[0])
        return (
            self.batched(observation),
            np.array([reward]),
            np.array([terminated]),
            np.array([truncated]),
            full_width(info),
        )

    def batched(self, observation: Any) -> Any:
        """A leading env axis of one, in a fresh array: a Roller carries an observation across a step,
        so a buffer reused between steps would rewrite the slot it already recorded.

        The observation space decides the structure, so a Dict observation widens leaf by leaf.
        """
        space = self.env.observation_space
        return concatenate(space, [observation], create_empty_array(space, n=1))
