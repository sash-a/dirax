"""The agent axis, which every field a stepper reports carries when an env has a team acting in it."""

from typing import Any, final

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


def over(terminated: Any, truncated: Any) -> bool:
    """Whether the episode is finished, from flags that may carry the agent axis.

    gymnasium's own `Autoreset` and `RecordEpisodeStatistics` ask ``terminated or truncated``, which
    raises the moment a flag arrives per agent. This is that question asked of an array, and it is
    the only reason this repo carries a copy of either.
    """
    return bool(np.all(np.asarray(terminated) | np.asarray(truncated)))


@final
class Agents(gym.Wrapper):
    """The agent axis on reward and done flags, whatever width the env reported them in.

    One expression covers every env with a team in it:

    ```
        np.broadcast_to(value, num_agents)
    ```

    A team sharing a reward or a flag broadcasts the one value across the axis; a team paid per
    agent is already that shape, so the broadcast is a no-op.

    An env with a single agent is not wrapped at all. Its fields have no agent axis, and giving one
    a width-one axis would only oblige every network written for it to produce that axis back.
    Applying this at ``num_agents=1`` is therefore refused rather than silently done.

    The observation passes through untouched. It is read by exactly one thing, and ADR-0008 makes
    unpacking it the algorithm's job, so a team's observation arrives at whatever shape its env
    gives it.

    Attributes:
        num_agents: How wide the axis is.
    """

    def __init__(self, env: gym.Env, num_agents: int) -> None:
        if num_agents < 2:
            raise ValueError(
                f"num_agents is {num_agents}, and an env with fewer than two agents has no axis to widen to. "
                "A stepper leaves a single-agent env unwrapped."
            )
        super().__init__(env)
        self.num_agents = num_agents

    def step(self, action: Any) -> tuple[Any, NDArray, NDArray, NDArray, dict[str, Any]]:  # ty: ignore[invalid-method-override]
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            observation,
            self.widened(reward).astype(np.float32),
            self.widened(terminated).astype(np.bool_),
            self.widened(truncated).astype(np.bool_),
            info,
        )

    def widened(self, value: Any) -> NDArray:
        """Add the agent dimension."""
        return np.broadcast_to(np.asarray(value), self.num_agents)


def one_agent(space: gym.Space, num_agents: int) -> gym.Space:
    """One agent's action space, out of the space naming the whole team's — what a Head is built from.

    Not a wrapper's job: nothing in the chain rewrites an action, so this is the caller answering the
    one question a Head asks, at the point they build it.
    """
    match space:
        case gym.spaces.Tuple(spaces=spaces) if len(spaces) == num_agents:
            return spaces[0]
        case gym.spaces.MultiDiscrete(nvec=nvec) if len(nvec) == num_agents:
            return gym.spaces.Discrete(int(nvec[0]))
        case gym.spaces.Box(shape=(agents, width)) if agents == num_agents:
            return gym.spaces.Box(space.low[0], space.high[0], (width,), space.low.dtype.type)
        case _ if num_agents == 1:
            return space
        case _:
            raise TypeError(f"{space} does not name {num_agents} agents' actions.")
