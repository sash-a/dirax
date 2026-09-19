"""The parameter path, asserted where a caller can see it rather than through a Trajectory.

A Policy reading stale parameters produces plausible Trajectories and no error anywhere, so the
Sync point, the key advance and the host boundary are pinned directly.
"""

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, overload

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brrrl.acting.policy import FFPolicy
from brrrl.params.params import ParamStore
from brrrl.types import Array, Params, PRNGKey, VersionedParams

OBSERVATION = jnp.float32(3.0)


class Output(NamedTuple):
    action: Array


@dataclass
class Scripted:
    """A Source handing back a scripted sequence, so a `get` that finds nothing new is reachable."""

    updates: list[VersionedParams | None]

    @overload
    def get(self, block: Literal[True]) -> VersionedParams: ...
    @overload
    def get(self, block: Literal[False]) -> VersionedParams | None: ...
    def get(self, block: bool) -> VersionedParams | None:
        """Nothing new, once the script runs out."""
        return self.updates.pop(0) if self.updates else None


def scaled(params: Params, observation: jax.Array, key: PRNGKey) -> Output:
    """Pure in the params, so which Version acted is readable off the action."""
    return Output(action=params["params"]["scale"] * observation)


def keyed(params: Params, observation: jax.Array, key: PRNGKey) -> Output:
    """Pure in the key, so which key acted is readable off the action."""
    return Output(action=jax.random.bits(key))


def seeded(scale: float = 2.0, version: int = 0) -> ParamStore:
    return ParamStore(VersionedParams({"params": {"scale": jnp.float32(scale)}}, version))


def acted(policy: FFPolicy[Any, Output]) -> float:
    return np.asarray(policy.act(OBSERVATION)[0].action).item()


def test_a_policy_acts_under_the_seeded_parameters_before_anything_is_published() -> None:
    """The driver seeds the Store, so nothing stalls at start-up waiting for a first publish."""
    policy = FFPolicy(scaled, seeded(scale=2.0), jax.random.key(0))

    output, version = policy.act(OBSERVATION)

    assert version == 0
    assert np.asarray(output.action) == pytest.approx(6.0)


def test_an_act_reports_the_version_of_the_parameters_behind_it() -> None:
    """Which is what lets a Roller stamp a Trajectory without ever asking a Policy to align."""
    store = seeded(scale=2.0)
    policy = FFPolicy(scaled, store, jax.random.key(0))

    store.publish({"params": {"scale": jnp.float32(10.0)}})
    output, version = policy.act(OBSERVATION)

    assert version == 1
    assert np.asarray(output.action) == pytest.approx(30.0)


def test_a_policy_adopts_new_parameters_as_soon_as_they_land() -> None:
    """When to align is the Policy's own business, and pulling at the act rate is the fastest a
    published update can reach a rollout: the step after it lands, not the Trajectory after."""
    store = seeded(scale=2.0)
    policy = FFPolicy(scaled, store, jax.random.key(0))

    before = acted(policy)
    store.publish({"params": {"scale": jnp.float32(10.0)}})

    assert before == pytest.approx(6.0)
    assert acted(policy) == pytest.approx(30.0)


def test_a_sync_point_that_finds_nothing_new_leaves_the_current_parameters_in_place() -> None:
    """A Sync point never blocks, so an Actor keeps producing while it waits for the next publish."""
    source = Scripted([VersionedParams({"params": {"scale": jnp.float32(2.0)}}, 7)])
    policy = FFPolicy(scaled, source, jax.random.key(0))

    assert policy.sync() == 7
    assert policy.sync() == 7
    assert acted(policy) == pytest.approx(6.0)


def test_act_returns_host_arrays() -> None:
    """A record function and the accumulator only ever see numpy, and an inference-server client
    could return nothing else."""
    policy = FFPolicy(scaled, seeded(), jax.random.key(0))

    action = policy.act(OBSERVATION)[0].action

    assert isinstance(action, np.ndarray)
    assert not isinstance(action, jax.Array)


def test_keys_advance_and_never_repeat() -> None:
    """A Roller supplies no key, so a Policy carrying its own is the only thing keeping the stream
    of one act independent of the last."""
    policy = FFPolicy(keyed, seeded(), jax.random.key(0))

    actions = [acted(policy) for _ in range(5)]

    assert len(set(actions)) == 5


def test_two_policies_seeded_differently_sample_differently() -> None:
    """Seeding is per Policy, so two Rollers on one Actor do not sample in lockstep by accident."""
    left = FFPolicy(keyed, seeded(), jax.random.key(0))
    right = FFPolicy(keyed, seeded(), jax.random.key(1))

    assert acted(left) != acted(right)
