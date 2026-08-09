"""The option algebra: what a claim, a Component and a reservation merge into.

ADR-0011 divides Ray's actor options three ways — the claim is the caller's, the behaviour
is the class's, identity and derived environment are the placement site's — and every rule
of that division is asserted here. None of it needs a cluster, because the merge is a pure
function of its three inputs.
"""

from collections.abc import Mapping
from dataclasses import fields
from typing import Any, ClassVar

import pytest
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from brrrl.distributed.placement.component import Behaviour
from brrrl.distributed.placement.placement import Placement, Placements, reserve

CLAIM: dict[str, Any] = {
    "num_cpus": 2,
    "num_gpus": 0.5,
    "memory": 8 << 30,
    "resources": {"nic": 1},
    "label_selector": {"role": "learner"},
    "accelerator_type": "H100",
}
"""One representative value per scheduling field, so a field added without one fails loudly below."""

RESERVABLE = set(CLAIM) - {"label_selector"}
"""Labels reach a reservation as a per-bundle selector rather than as a bundle resource."""


class Bare:
    """A Component whose correctness asks nothing of Ray."""

    RAY_OPTIONS: ClassVar[Behaviour] = {}


class Ordered:
    """The CheckpointStore's shape: fire-and-forget saves rest on Ray executing calls in order."""

    RAY_OPTIONS: ClassVar[Behaviour] = {"allow_out_of_order_execution": False, "max_task_retries": 0}


def test_an_unset_claim_asks_the_scheduler_for_nothing() -> None:
    """``Placements()`` has to be a runnable configuration, so unset fields are dropped, not sent as None."""
    options = Placement().options(Bare)

    assert [key for key in CLAIM if key in options] == []


def test_a_claim_reaches_the_scheduler_under_rays_own_names() -> None:
    """Every scheduling field is named after the Ray option it becomes, so the merge does no translating."""
    options = Placement(**CLAIM).options(Bare)

    assert {key: options[key] for key in CLAIM} == CLAIM


def test_every_claim_field_is_either_scheduled_or_the_environment() -> None:
    """Forces a new field to be classified rather than silently ignored by the merge."""
    assert {field.name for field in fields(Placement)} == set(CLAIM) | {"env"}


@pytest.mark.parametrize("field", sorted(RESERVABLE))
def test_every_reservable_claim_field_reaches_the_bundle(field: str) -> None:
    """A field the scheduler honours but the bundle omits means a Gang member does not fit the
    bundle reserved for it, and hangs — the mismatch deriving the bundle exists to prevent."""
    assert Placement(**{field: CLAIM[field]}).bundle() != {}


def test_a_components_behaviour_is_merged_in() -> None:
    """The class's half of the division: options a config file must not be able to reach."""
    options = Placement().options(Ordered)

    assert options["allow_out_of_order_execution"] is False
    assert options["max_task_retries"] == 0


def test_claiming_no_gpu_derives_a_cpu_only_jax_platform() -> None:
    """Ray leaves ``CUDA_VISIBLE_DEVICES`` unset — not blank — for a process claiming no GPU.

    Without this, every Component on a GPU box initialises CUDA and preallocates most of a card
    it was never given, and the first to touch JAX wins.
    """
    assert env_of(Placement().options(Bare)) == {"JAX_PLATFORMS": "cpu"}
    assert env_of(Placement(num_gpus=0).options(Bare)) == {"JAX_PLATFORMS": "cpu"}


def test_claiming_a_whole_gpu_derives_nothing() -> None:
    """The card is the process's alone, so JAX's own defaults are right and no runtime environment is needed."""
    assert "runtime_env" not in Placement(num_gpus=1).options(Bare)


def test_claiming_a_fraction_of_a_gpu_derives_a_matching_memory_fraction() -> None:
    """Otherwise a Trainer and an inference server sharing one card each try to take 75% of it."""
    assert env_of(Placement(num_gpus=0.25).options(Bare)) == {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.25"}


def test_the_callers_environment_wins_over_a_derived_variable() -> None:
    """An unusual accelerator setup must not be fenced off by the library's inference."""
    claim = Placement(num_gpus=0.5, env={"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9", "MUJOCO_GL": "egl"})

    assert env_of(claim.options(Bare)) == {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9", "MUJOCO_GL": "egl"}


def test_a_reservation_becomes_a_scheduling_strategy_carrying_the_members_index() -> None:
    """A Gang member's bundle index is its stable identity, and the direct ``placement_group=``
    options Ray once took for this are deprecated."""
    group = PlacementGroup.empty()

    strategy = Placement().options(Bare, group, index=2)["scheduling_strategy"]

    assert isinstance(strategy, PlacementGroupSchedulingStrategy)
    assert strategy.placement_group is group
    assert strategy.placement_group_bundle_index == 2


def test_no_reservation_leaves_the_scheduling_strategy_to_ray() -> None:
    assert "scheduling_strategy" not in Placement().options(Bare)


def test_a_bundle_is_the_claim_itself() -> None:
    """Deriving it is the point: state a bundle separately and a mismatch means the member
    does not fit the bundle reserved for it, and hangs."""
    assert Placement(**CLAIM).bundle() == {
        "CPU": 2,
        "GPU": 0.5,
        "memory": 8 << 30,
        "nic": 1,
        "accelerator_type:H100": 0.001,
    }


def test_an_unset_claim_has_nothing_to_reserve() -> None:
    """Ray refuses an empty bundle, so a Gang with no claim is a configuration error, not a hang."""
    assert Placement().bundle() == {}


def test_every_role_is_a_field_and_defaults_to_an_unset_placement() -> None:
    """An open mapping would leave a misspelled role unplaced with no error — the silence this removes."""
    placements = Placements()

    assert {field.name for field in fields(placements)} == {
        "trainer",
        "actor",
        "inference_server",
        "params",
        "metrics",
        "checkpoints",
    }
    assert all(getattr(placements, field.name) == Placement() for field in fields(placements))


def test_reserving_a_claim_that_asks_for_nothing_is_refused_rather_than_left_to_hang() -> None:
    with pytest.raises(ValueError, match="asks for no resources"):
        reserve(Placement(), size=2)


def env_of(options: dict[str, Any]) -> Mapping[str, str]:
    return options["runtime_env"]["env_vars"]
