from collections.abc import Callable, Iterator
from functools import partial
from typing import Any, Protocol, final

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, VectorEnv

from brrrl.types import StepResult
from brrrl.wrappers.agents import Agents
from brrrl.wrappers.autoreset import Autoreset
from brrrl.wrappers.looped import Looped
from brrrl.wrappers.statistics import EpisodeStatistics, Reduction

# What a stepper applies itself, and so must not be already present in the env
OWNED_WRAPPERS = (
    gym.wrappers.Autoreset,
    Autoreset,
    Agents,
    gym.wrappers.RecordEpisodeStatistics,
    EpisodeStatistics,
)

# Signature for the wrapping function a stepper applies over its vectorised envs
type VectorWrap = Callable[[VectorEnv], VectorEnv]


def same[E](env: E) -> E:
    """The wrap a stepper takes when a caller names none."""
    return env


class EnvStepper[Obs, Act](Protocol):
    """The gymnasium step API, with a leading env axis and an agent axis where there is a team.

    Reward, both flags and every field derived from them arrive at ``(E, A, ...)`` for an env with a
    team acting in it, and at ``(E, ...)`` for one with a single agent, which has no agent axis at
    all. `prefix` is that leading shape, and the only place the arity is asked.

    A width-one axis would be the tidier contract here, and it was the one this had. It cost more
    than it bought: nothing downstream branches on the arity either way — a Learner's arithmetic is
    polymorphic over leading axes, the same property that lets acting at ``(E, ...)`` and learning at
    ``(T, E, ...)`` call one network — while a width-one axis obliged every network written for a
    single agent to produce an agent axis nobody had asked about (ADR-0012).

    Beside the step it reports its finished episodes in the vector episode-statistics schema:
    full-width ``r``, ``l`` and ``t`` under ``episode``, and the boolean mask naming which envs
    finished under ``_episode``. That contract is what lets a Roller read episodes the same way
    whatever built the environments.
    """

    @property
    def num_envs(self) -> int: ...

    @property
    def num_agents(self) -> int: ...

    def reset(self, *, seed: int | None = None) -> tuple[Obs, dict[str, Any]]: ...

    def step(self, action: Act) -> StepResult[Obs]: ...

    def close(self) -> None:
        """Release the environments. Part of the trait because a stepper is free to hold processes,
        and nothing above it can know whether the one it was handed does."""
        ...


@final
class LoopedStepper[Obs, Act]:
    """One env per maker, stepped in turn, presented as an `EnvStepper`.

    Envs run in a loop on a single process, not in parallel.
    """

    def __init__(
        self,
        env_makers: list[Callable[[], gym.Env]],
        num_agents: int = 1,
        wrap: VectorWrap = same,
        reduce: Reduction = np.sum,
    ) -> None:
        """Build one env per maker, and hand them all to `Looped`.

        Args:
            env_makers: One factory per env, so how many there are is the env axis.
            num_agents: How wide the agent axis is, one for an env with a single agent.
            wrap: Applied to the vectorised envs, for a transform that wants them all at once.
            reduce: How an episode's per-agent returns become the number a run is judged on.
        """
        self.num_agents = num_agents
        self.env = wrap(Looped([built(make_env, num_agents, reduce) for make_env in counted(env_makers)]))

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def reset(self, *, seed: int | None = None) -> tuple[Obs, dict[str, Any]]:
        return self.env.reset(seed=seed)

    def step(self, action: Act) -> StepResult[Obs]:
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()


@final
class AsyncStepper[Obs, Act]:
    """``num_envs`` envs stepped on their own processes, presented as an `EnvStepper`."""

    def __init__(
        self,
        env_makers: list[Callable[[], gym.Env]],
        num_agents: int = 1,
        wrap: VectorWrap = same,
        reduce: Reduction = np.sum,
        shared_memory: bool = True,
        context: str | None = "forkserver",
    ) -> None:
        """Build one env per maker, one per process, and hand the result to ``wrap``.

        Args:
            env_makers: One factory per env. Each is called in its own worker, and shipped there by
                cloudpickle, so a closure is fine and anything it captures has to survive the trip.
            num_agents: How wide the agent axis is, one for an env with a single agent.
            wrap: Applied to the vectorised envs, for a transform that wants them all at once.
            reduce: How an episode's per-agent returns become the number a run is judged on.
            shared_memory: Pass observations back through shared memory rather than the pipe.
            context: How the workers are started.
        """
        self.num_agents = num_agents
        self.env = wrap(
            AsyncVectorEnv(
                [partial(built, make_env, num_agents, reduce) for make_env in counted(env_makers)],
                shared_memory=shared_memory,
                autoreset_mode=AutoresetMode.DISABLED,
                context=context,
            )
        )

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def reset(self, *, seed: int | None = None) -> tuple[Obs, dict[str, Any]]:
        return self.env.reset(seed=seed)

    def step(self, action: Act) -> StepResult[Obs]:
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()


def prefix(num_envs: int, num_agents: int) -> tuple[int, ...]:
    """The leading shape every field a stepper reports carries: ``(E, A)`` for a team, ``(E,)`` alone.

    The one place the arity is asked, so that a caller sizing a buffer against a stepper composes a
    shape rather than answering the question again.
    """
    return (num_envs, num_agents) if num_agents > 1 else (num_envs,)


def required_wrappers(env: gym.Env, num_agents: int, reduce: Reduction = np.sum) -> gym.Env:
    """``env`` with everything a stepper applies itself, in the order its guarantees need.

    `Agents` goes on only where there is a team to widen to. A single agent's env is left with the
    scalar reward and bare flags a `gym.Env` reports, which the vectorisation above stacks into the
    ``(E,)`` a Roller reads.
    """
    wrapped = Autoreset(EpisodeStatistics(steppable(env), reduce))
    return Agents(wrapped, num_agents) if num_agents > 1 else wrapped


def built(make_env: Callable[[], gym.Env], num_agents: int, reduce: Reduction = np.sum) -> gym.Env:
    """`required_wrappers` around the env ``make_env`` builds, where a stepper has a maker not an env."""
    return required_wrappers(make_env(), num_agents, reduce)


def counted(env_makers: list[Callable[[], gym.Env]]) -> list[Callable[[], gym.Env]]:
    """``env_makers``, refusing an env axis with nothing on it."""
    if not env_makers:
        raise ValueError("env_makers is empty, and a stepper with no environments steps nothing.")
    return env_makers


def wrappers_of(env: gym.Env) -> Iterator[type[gym.Wrapper]]:
    """Every wrapper already around ``env``, outermost first."""
    while isinstance(env, gym.Wrapper):
        yield type(env)
        env = env.env


def steppable(env: gym.Env) -> gym.Env:
    """``env``, refusing a wrapper a stepper applies itself and would otherwise apply twice."""
    if isinstance(env, VectorEnv):
        raise TypeError(
            f"{type(env).__name__} is already vectorised, and a stepper vectorises what it is given. "
            "Pass a factory for one env per env you want."
        )
    if owns := [wrapper.__name__ for wrapper in wrappers_of(env) if issubclass(wrapper, OWNED_WRAPPERS)]:
        raise ValueError(f"{owns} already wraps this env, and a stepper applies that itself. Pass the env without it.")
    return env
