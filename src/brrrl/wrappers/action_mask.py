from collections.abc import Callable
from typing import Any, SupportsFloat, final

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

OBSERVATION = "observation"
ACTION_MASK = "action_mask"
"""The keys an observation and its legal actions are read under, once the two travel together."""

type MaskFn = Callable[[dict[str, Any]], NDArray[np.bool_]]
"""Reads the legal actions out of what an env reported beside a step."""


def published(info: dict[str, Any]) -> NDArray[np.bool_]:
    """The key gymnasium's own masked envs report a mask under, which is the only convention there is."""
    if ACTION_MASK not in info:
        raise KeyError(f"this env reports no {ACTION_MASK!r} beside its steps; pass a reader that knows where it does")
    return info[ACTION_MASK]


@final
class ActionMask(gym.Wrapper):
    """An observation carrying which of its actions are legal, beside the observation itself.

    Only an env that masks something wants this. A run whose env masks nothing carries no mask at
    all rather than an all-true one, and the Assembly reads a plain observation: the widths add up
    on Atari, and a mask nothing ever reads is a leaf on every hop a Trajectory makes.

    Every state has to leave at least one action legal, terminal ones included. A state that leaves
    none has an undefined policy, and the nan that follows reaches the gradient through slots a loss
    has already masked off — so it is an env to fix rather than a case to handle.

    Attributes:
        legal: Reads the mask out of an env's info, defaulting to the key gymnasium reports it under.
    """

    def __init__(self, env: gym.Env, legal: MaskFn = published) -> None:
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError(f"{env.action_space} has no actions to mask off, which only a discrete space has.")
        super().__init__(env)
        self.legal = legal
        self.observation_space = gym.spaces.Dict(
            {OBSERVATION: env.observation_space, ACTION_MASK: gym.spaces.MultiBinary(int(env.action_space.n))}
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        return self.masked(observation, info), info

    def step(self, action: Any) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        """One env's own step shape, scalars and all: the widening that makes it a stepper's is above."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.masked(observation, info), reward, terminated, truncated, info

    def masked(self, observation: Any, info: dict[str, Any]) -> dict[str, Any]:
        """The mask reported beside this observation, so the two describe the same state.

        Read out of the info rather than off the env, because that is where an env that has one puts
        it, and re-deriving it from the env's own state means reaching past every wrapper to do it.
        """
        return {OBSERVATION: observation, ACTION_MASK: self.legal(info).astype(np.int8)}
