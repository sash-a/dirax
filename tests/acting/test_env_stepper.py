"""One stepper family, two ways of running it, and one contract between them.

Every field arrives at ``(E, A, ...)`` where a team acts and ``(E, ...)`` where a single agent does,
whatever built the environments. The agent axis is one an env with one agent does not have, rather
than one it fills at width one: nothing downstream branches on the arity either way, because a
Learner's arithmetic is polymorphic over leading axes, while a width-one axis obliged every network
written for a single agent to produce an axis nobody had asked about (ADR-0012).

What differs between the two steppers is only where the envs run: both take the same vector wrap,
applied outside the vectorisation, because both hold a real vector env by the time a caller's
transform goes on. One env is not a third way — it is `LoopedStepper` handed one maker, which is the
whole reason there is no stepper of its own for it.

ADR-0001 chose next-step autoreset because same-step returns an observation belonging to a different
episode than the flag beside it. That discipline is ours on every path now rather than gymnasium's
on some, so the refusals that keep a second copy of it off an env matter more, not less. The stream
it produces is held to gymnasium's own vector env in ``test_rollout.py``, which is the oracle the
layout was settled against.
"""

from contextlib import closing
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.wrappers.vector import NormalizeReward

from brrrl.acting.env_stepper import AsyncStepper, LoopedStepper, prefix
from brrrl.wrappers.agents import Agents
from brrrl.wrappers.autoreset import Autoreset
from brrrl.wrappers.looped import Looped
from brrrl.wrappers.statistics import ENV_STATS, EPISODE_MASK, EPISODE_STATS, EpisodeStatistics, env_stats

NUM_ENVS, AGENTS, LENGTH, CARTPOLE_OBS = 2, 3, 5, 4
SHAPED = 2.5
"""What `Instrumented` pays itself, so an env's own metric is assertable rather than recorded."""
PAY = (1.0, 2.0, 3.0)
"""What each agent is paid a step, so a team's episode return is assertable rather than recorded."""


class Team(gym.Env):
    """A cooperative env in the shape rware and lbforaging report in: paid per agent, over at once."""

    def __init__(self, num_agents: int = AGENTS) -> None:
        self.observation_space = gym.spaces.Box(0.0, 1.0, (2,), np.float32)
        self.action_space = gym.spaces.MultiDiscrete([2] * num_agents)
        self.num_agents = num_agents
        self.elapsed = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict]:
        self.elapsed = 0
        return np.zeros(2, np.float32), {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict]:
        self.elapsed += 1
        over = self.elapsed >= LENGTH
        return np.zeros(2, np.float32), PAY[: self.num_agents], (over,) * self.num_agents, False, {}


def cartpole() -> gym.Env:
    return gym.make("CartPole-v1")


def cartpoles() -> gym.vector.VectorEnv:
    """The factory a caller reaches for out of habit, which a stepper has to refuse."""
    return gym.make_vec("CartPole-v1", num_envs=NUM_ENVS)


def single(make_env=cartpole, num_agents=1, **kwargs) -> LoopedStepper:
    """One env, which is the family's debuggable end rather than a stepper of its own."""
    return LoopedStepper([make_env], num_agents, **kwargs)


def loop(make_env=cartpole, num_agents=1, **kwargs) -> LoopedStepper:
    return LoopedStepper([make_env] * NUM_ENVS, num_agents, **kwargs)


def processes(make_env=cartpole, num_agents=1, **kwargs) -> AsyncStepper:
    return AsyncStepper([make_env] * NUM_ENVS, num_agents, **kwargs)


VECTOR = [pytest.param(loop, id="loop"), pytest.param(processes, id="processes")]
EVERY = [pytest.param(single, id="single"), *VECTOR]


def idle(stepper) -> np.ndarray:
    """The do-nothing action, at the shape the stepper takes it in."""
    return np.zeros(prefix(stepper.num_envs, stepper.num_agents), int)


def rewarded(stepper) -> np.ndarray:
    """What one step pays, from a fresh reset."""
    stepper.reset()
    return stepper.step(idle(stepper))[1]


def returns(stepper, steps: int) -> list[np.ndarray]:
    """Every episode return the stepper reported over ``steps`` steps."""
    stepper.reset()
    action = idle(stepper)
    return [
        info[EPISODE_STATS]["r"][info[EPISODE_MASK]]
        for _ in range(steps)
        if EPISODE_STATS in (info := stepper.step(action)[4])
    ]


@pytest.mark.parametrize("build", EVERY)
@pytest.mark.parametrize(
    ("make_env", "num_agents", "axis"),
    [(cartpole, 1, ()), (Team, AGENTS, (AGENTS,))],
    ids=["one agent", "a team"],
)
def test_every_field_leads_with_an_env_axis_and_an_agent_axis_where_there_is_a_team(
    build, make_env, num_agents: int, axis: tuple[int, ...]
) -> None:
    """The one API a Roller steps. A single agent's fields have no agent axis at all, so a network
    written for one is handed and asked for nothing an agent count would show up in."""
    with closing(build(make_env, num_agents)) as stepper:
        stepper.reset()
        _, reward, terminated, truncated, _ = stepper.step(idle(stepper))

        assert stepper.num_agents == num_agents
        assert reward.shape == terminated.shape == truncated.shape == (stepper.num_envs, *axis)


@pytest.mark.parametrize("build", EVERY)
def test_a_reward_paid_per_agent_crosses_every_stepper(build) -> None:
    """The thing gymnasium's `SyncVectorEnv` could not do: it assigns each env's reward into a scalar
    slot of an ``(E,)`` buffer, so a per-agent reward cannot pass through it at all. None of these
    holds such a buffer — the axis goes on before the envs are stacked, never after."""
    with closing(build(Team, AGENTS)) as stepper:
        assert rewarded(stepper).tolist() == [list(PAY)] * stepper.num_envs


@pytest.mark.parametrize("build", VECTOR)
def test_one_env_is_built_per_maker(build) -> None:
    """How many makers there are is the env axis, which is the second factor in a Roller's transition
    count — so only the stepper has it."""
    with closing(build()) as stepper:
        assert stepper.num_envs == NUM_ENVS


@pytest.mark.parametrize("build", VECTOR)
def test_an_already_vectorised_env_is_refused_at_construction(build) -> None:
    """A stepper is handed what to vectorise. Something already vectorised carries an autoreset mode
    it no longer inspects, which is exactly the assumption ADR-0001 refuses to rest on."""
    with pytest.raises(TypeError, match="already vectorised"):
        build(cartpoles)


@pytest.mark.parametrize("stepper", [LoopedStepper, AsyncStepper], ids=["loop", "processes"])
def test_a_stepper_with_no_environments_is_refused(stepper) -> None:
    with pytest.raises(ValueError, match="no environments"):
        stepper([])


@pytest.mark.parametrize("build", EVERY)
@pytest.mark.parametrize(
    "owned",
    [gym.wrappers.Autoreset, Autoreset, lambda env: Agents(env, AGENTS), EpisodeStatistics],
    ids=["gymnasium's autoreset", "ours", "the agent axis", "the statistics"],
)
def test_an_env_that_already_carries_a_stepper_owned_wrapper_is_refused(build, owned) -> None:
    """Each of these is the stepper's own. A second autoreset inserts an extra ignored step and says
    nothing about it; a second agent axis widens a field that is already wide; a second recorder
    reports the wrong episode."""
    with pytest.raises(ValueError, match="applies that itself"):
        build(lambda: owned(cartpole()))


def test_the_agent_axis_is_not_put_on_an_env_with_a_single_agent() -> None:
    """The one condition the arity gets to make, and it is here. Everything above reads the shape a
    stepper reports rather than asking how many agents produced it."""
    with closing(single()) as stepper:
        assert not any(isinstance(wrapper, Agents) for wrapper in wrappers_of(inner(stepper)))

    with closing(single(Team, AGENTS)) as stepper:
        assert any(isinstance(wrapper, Agents) for wrapper in wrappers_of(inner(stepper)))


class Instrumented(gym.Wrapper):
    """An env reporting a metric of its own, which only the env that owns it can name."""

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info | {ENV_STATS: {"shaped": SHAPED}}


def instrumented() -> gym.Env:
    return Instrumented(cartpole())


@pytest.mark.parametrize("build", VECTOR)
def test_an_envs_own_metrics_read_the_same_whichever_vectorisation_built_them(build) -> None:
    """The property the whole schema is for. These two aggregate their infos differently, and a
    metric that survived one and not the other would be a curve that changed shape with a placement
    decision — the reader would have no way to tell."""
    with closing(build(instrumented)) as stepper:
        stepper.reset()
        infos = [stepper.step(idle(stepper))[4] for _ in range(3)]

    assert np.asarray(env_stats(infos)["shaped"]).tolist() == [SHAPED] * 3 * NUM_ENVS


def inner(stepper: LoopedStepper) -> gym.Env:
    """The env a looped stepper built, which is where the wrappers it applies itself went."""
    assert isinstance(stepper.env, Looped)
    return stepper.env.envs[0]


def wrappers_of(env: gym.Env):
    """Every wrapper around ``env``, outermost first."""
    while isinstance(env, gym.Wrapper):
        yield env
        env = env.env


def test_widening_an_env_with_one_agent_is_refused() -> None:
    """`Agents` exists to broadcast a team's fields across an axis. At one agent there is no axis to
    widen to, and adding one would oblige every network written for that env to produce it back."""
    with pytest.raises(ValueError, match="fewer than two agents"):
        Agents(cartpole(), 1)


@pytest.mark.parametrize("build", EVERY)
def test_an_episode_return_is_reported_in_the_environments_own_units(build) -> None:
    """The statistics are taken under the scaling. Above it, every run's returns would be divided by
    a spread that moves as the policy does, and no two runs could be compared at all."""
    with closing(build(wrap=NormalizeReward)) as stepper:
        reported = returns(stepper, steps=600)

        assert reported, "no episode finished, so the assertion below would hold vacuously"
        assert all(finished == int(finished) for finished in np.concatenate(reported))


@pytest.mark.parametrize("build", EVERY)
def test_a_teams_episode_return_is_what_its_agents_were_paid(build) -> None:
    """One number in the end, and by default the sum of the agents' shares — which is what rware and
    lbforaging's kind of env pays out."""
    with closing(build(Team, AGENTS)) as stepper:
        reported = returns(stepper, steps=LENGTH)

        assert np.concatenate(reported).tolist() == [sum(PAY) * LENGTH] * stepper.num_envs


@pytest.mark.parametrize("build", EVERY)
def test_the_reduction_reaches_the_recorder_underneath(build) -> None:
    """A team's returns become one number somewhere, and no default covers both an env paying per
    agent and one broadcasting a shared reward. A stepper carries the choice rather than making it."""
    with closing(build(Team, AGENTS, reduce=np.mean)) as stepper:
        reported = returns(stepper, steps=LENGTH)

        assert np.concatenate(reported).tolist() == [pytest.approx(sum(PAY) * LENGTH / AGENTS)] * stepper.num_envs


@pytest.mark.parametrize("build", EVERY)
def test_a_wrap_is_handed_every_env_the_stepper_built(build) -> None:
    """It goes on outside the vectorisation, so scaling across every env at once needs nothing but a
    stepper — and what it is handed is already autoreset and already recorded, so neither of the
    stepper's own guarantees is on offer to break.

    CartPole pays exactly one a step, so anything but one is the caller's wrapper and nothing else.
    What it scales is what a Learner sees, and never what an episode return is reported in."""
    with closing(build()) as plain, closing(build(wrap=NormalizeReward)) as scaled:
        assert np.all(rewarded(plain) == 1.0)
        assert not np.any(rewarded(scaled) == 1.0)


@pytest.mark.parametrize("build", EVERY)
def test_a_wrap_written_for_one_env_is_refused_by_the_axes_it_cannot_carry(build) -> None:
    """The seam moved outside the vectorisation, so gymnasium's own per-env `NormalizeReward` is not
    something a caller can name here: it reads ``float(reward)`` of what is now at least an ``(E,)``
    array. Its vector counterpart is the one that belongs, and for a single agent it is the whole
    reason this package owns no reward wrapper of its own."""
    with pytest.raises((TypeError, AssertionError, ValueError)):
        build(wrap=gym.wrappers.NormalizeReward)


def test_a_loop_over_one_env_is_an_ordinary_loop() -> None:
    """Debugging one env exercises the same code path as a hundred, and the vector statistics schema
    needs no hand-synthesising."""
    with closing(LoopedStepper([cartpole])) as stepper:
        observation, _ = stepper.reset()

        assert stepper.num_envs == 1
        assert observation.shape == (1, CARTPOLE_OBS)


def test_an_observation_is_a_fresh_array_each_step() -> None:
    """A Roller carries an observation across a step, so a buffer reused between steps would rewrite
    the slot it already recorded — with the *next* state, under the flags of this one."""
    with closing(loop()) as stepper:
        first, _ = stepper.reset()
        kept = first.copy()
        for _ in range(3):
            stepper.step(idle(stepper))

        assert np.array_equal(first, kept)
