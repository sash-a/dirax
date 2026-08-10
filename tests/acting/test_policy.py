"""The parameter path, asserted where a caller can see it rather than through a Trajectory.

A Policy reading stale parameters produces plausible Trajectories and no error anywhere, so the
Sync point, the key advance and the host boundary are pinned directly.
"""

from dataclasses import dataclass
from typing import Literal, NamedTuple, overload

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brrrl.acting.params import InMemoryParamStore
from brrrl.acting.policy import LocalPolicy
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
    def get(self, block: Literal[False] = False) -> VersionedParams | None: ...
    def get(self, block: bool = False) -> VersionedParams | None:
        return self.updates.pop(0)


def scaled(params: Params, observation: jax.Array, key: PRNGKey) -> Output:
    """Pure in the params, so which Version acted is readable off the action."""
    return Output(action=params["params"]["scale"] * observation)


def keyed(params: Params, observation: jax.Array, key: PRNGKey) -> Output:
    """Pure in the key, so which key acted is readable off the action."""
    return Output(action=jax.random.bits(key))


def seeded(scale: float = 2.0, version: int = 0) -> InMemoryParamStore:
    return InMemoryParamStore(VersionedParams({"params": {"scale": jnp.float32(scale)}}, version))


def acted(policy: LocalPolicy[jax.Array, Output]) -> float:
    return np.asarray(policy.act(OBSERVATION).action).item()


def test_the_write_side_assigns_the_version() -> None:
    """One monotonic authority for a run: a publisher that passed its own count could repeat one."""
    store = seeded()

    assert store.publish({"params": {"scale": jnp.float32(3.0)}}) == 1
    assert store.publish({"params": {"scale": jnp.float32(4.0)}}) == 2


def test_a_resumed_run_continues_its_version_count() -> None:
    """Staleness stays comparable across a restart, so the driver seeds the count it left off at."""
    assert seeded(version=41).publish({"params": {"scale": jnp.float32(3.0)}}) == 42


def test_a_policy_acts_under_the_seeded_parameters_before_anything_is_published() -> None:
    """The driver seeds the Store, so nothing stalls at start-up waiting for a first publish."""
    policy = LocalPolicy(scaled, seeded(scale=2.0), jax.random.key(0))

    assert policy.sync() == 0
    assert acted(policy) == pytest.approx(6.0)


def test_new_parameters_are_adopted_only_at_a_sync_point() -> None:
    """A Trajectory carries exactly one Version, which is only true if acting never adopts."""
    store = seeded(scale=2.0)
    policy = LocalPolicy(scaled, store, jax.random.key(0))
    policy.sync()

    store.publish({"params": {"scale": jnp.float32(10.0)}})

    assert acted(policy) == pytest.approx(6.0)
    assert policy.sync() == 1
    assert acted(policy) == pytest.approx(30.0)


def test_a_sync_point_that_finds_nothing_new_leaves_the_current_parameters_in_place() -> None:
    """A Sync point never blocks, so an Actor keeps producing while it waits for the next publish."""
    source = Scripted([VersionedParams({"params": {"scale": jnp.float32(2.0)}}, 7), None, None])
    policy = LocalPolicy(scaled, source, jax.random.key(0))

    assert policy.sync() == 7
    assert policy.sync() == 7
    assert acted(policy) == pytest.approx(6.0)


def test_act_returns_host_arrays() -> None:
    """A record function and the accumulator only ever see numpy, and an inference-server client
    could return nothing else."""
    policy = LocalPolicy(scaled, seeded(), jax.random.key(0))

    action = policy.act(OBSERVATION).action

    assert isinstance(action, np.ndarray)
    assert not isinstance(action, jax.Array)


def test_keys_advance_and_do_not_repeat_across_a_refill() -> None:
    """Minted in bulk — one split per refill rather than one per step — and still never reused."""
    policy = LocalPolicy(keyed, seeded(), jax.random.key(0), keys_per_refill=2)

    actions = [acted(policy) for _ in range(5)]

    assert len(set(actions)) == 5


def test_two_policies_seeded_differently_sample_differently() -> None:
    """Seeding is per Policy, so two Rollers on one Actor do not sample in lockstep by accident."""
    left = LocalPolicy(keyed, seeded(), jax.random.key(0))
    right = LocalPolicy(keyed, seeded(), jax.random.key(1))

    assert acted(left) != acted(right)
