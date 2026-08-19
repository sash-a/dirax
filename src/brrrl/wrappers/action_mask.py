from collections.abc import Callable
from typing import Any, SupportsFloat, final

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

OBSERVATION = "observation"
ACTION_MASK = "action_mask"

type MaskFn = Callable[[dict[str, Any]], NDArray[np.bool_]]


def published(info: dict[str, Any]) -> NDArray[np.bool_]:
    """The key gymnasium's own masked envs report a mask in infos, this reads it."""
    if ACTION_MASK not in info:
        raise KeyError(f"this env reports no {ACTION_MASK!r} in infos. Either add a mask to infos or change mask_fn")
    return info[ACTION_MASK]


def legal_shape(space: gym.Space) -> tuple[int, ...]:
    """``(agents, actions)`` — one row of legal actions per agent, or ``(actions,)`` for one agent.

    The mask carries the agent axis wherever the reward does, and for the same reason: a Head reads
    its logits at that shape, so a mask a row short would broadcast against them rather than fail.
    An env with a single agent has no agent axis, so neither has its mask.
    """
    match space:
        case gym.spaces.Discrete(n=actions):
            return (int(actions),)
        case gym.spaces.MultiDiscrete(nvec=nvec) if nvec.ndim == 1 and len(np.unique(nvec)) == 1:
            return len(nvec), int(nvec[0])
        case gym.spaces.Tuple(spaces=spaces) if actions := alike(spaces):
            return len(spaces), actions
        case _:
            raise TypeError(f"{space} has no actions to mask off, which only a discrete space has.")


def alike(spaces: tuple[gym.Space, ...]) -> int:
    """How many actions every agent chooses between, or none if they do not all choose alike.

    A ragged mask has no rectangle to be, and padding one invents legal actions that do not exist.
    """
    discrete = [one for one in spaces if isinstance(one, gym.spaces.Discrete)]
    if not discrete or len(discrete) != len(spaces) or len({int(one.n) for one in discrete}) != 1:
        return 0
    return int(discrete[0].n)


@final
class ActionMask(gym.Wrapper):
    """An observation carrying which of its actions are legal, beside the observation itself.

    Attributes:
        mask_fn: Reads the mask out of an env's info, defaulting to the key gymnasium reports it under.
        shape: The shape every mask this wrapper carries is read at — `legal_shape` of the action space.
    """

    def __init__(self, env: gym.Env, mask_fn: MaskFn = published) -> None:
        super().__init__(env)
        self.mask_fn = mask_fn
        self.shape = legal_shape(env.action_space)
        self.observation_space = gym.spaces.Dict(
            {OBSERVATION: env.observation_space, ACTION_MASK: gym.spaces.MultiBinary(self.shape)}
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
        return {OBSERVATION: observation, ACTION_MASK: self.mask_fn(info).astype(np.int8).reshape(self.shape)}
