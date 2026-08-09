from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import ray
from ray.actor import ActorClass
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from brrrl.distributed.placement.component import Component

type Strategy = Literal["PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"]


@dataclass(frozen=True, slots=True)
class Placement:
    """What one role's processes claim, and where they may land.

    Every field is unset by default, so a run naming no Placement lands wherever Ray will
    take it — which is what makes a laptop need no configuration and no cluster detection.
    The set is closed: the options it excludes are the ones a caller could use to break a
    Component's own correctness.
    """

    num_cpus: float | None = None
    num_gpus: float | None = None
    memory: int | None = None
    resources: Mapping[str, float] | None = None
    label_selector: Mapping[str, str] | None = None
    accelerator_type: str | None = None
    env: Mapping[str, str] | None = None

    def options(
        self, component: type[Component], group: PlacementGroup | None = None, index: int = -1
    ) -> dict[str, Any]:
        """This claim, ``component``'s behaviour and what the placement site derives, in that order.

        Args:
            component: The class being placed, which supplies the behaviour its correctness needs.
            group: A reservation to place into, if this Component is a member of a Gang.
            index: The member's bundle in that reservation, which is its stable identity.
        """
        claim = {
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "memory": self.memory,
            "resources": self.resources,
            "label_selector": self.label_selector,
            "accelerator_type": self.accelerator_type,
        }
        environment = {**self.jax_env(), **(self.env or {})}
        derived: dict[str, Any] = {}
        if group is not None:
            derived["scheduling_strategy"] = PlacementGroupSchedulingStrategy(group, index)
        if environment:
            derived["runtime_env"] = {"env_vars": environment}
        return {key: value for key, value in claim.items() if value is not None} | dict(component.RAY_OPTIONS) | derived

    def apply(self, component: type[Component], group: PlacementGroup | None = None, index: int = -1) -> ActorClass:
        """``component`` as a Ray actor class carrying these options, ready to construct.

        Split from ``options`` because Ray's option wrapper is not introspectable: the mapping is
        the testable half, and this is the line that hands it over.
        """
        return ray.remote(component).options(**self.options(component, group, index))

    def bundle(self) -> dict[str, float]:
        """The claim as a reservation bundle, so a Gang member fits its own bundle by construction."""
        named = {"CPU": self.num_cpus, "GPU": self.num_gpus, "memory": self.memory}
        claimed = {key: value for key, value in named.items() if value is not None}
        accelerator = {f"accelerator_type:{self.accelerator_type}": 0.001} if self.accelerator_type else {}
        return claimed | dict(self.resources or {}) | accelerator

    def jax_env(self) -> dict[str, str]:
        """The JAX environment this claim implies, which the caller's ``env`` then overrides.

        Ray leaves ``CUDA_VISIBLE_DEVICES`` unset rather than blank for a process claiming no
        GPU, so a process that says nothing preallocates most of a card it was never given.
        """
        if not self.num_gpus:
            return {"JAX_PLATFORMS": "cpu"}
        if self.num_gpus < 1:
            return {"XLA_PYTHON_CLIENT_MEM_FRACTION": str(self.num_gpus)}
        return {}


@dataclass(frozen=True, slots=True)
class Placements:
    """One Placement per role brrrl defines, and no way to name a role it does not.

    Closed rather than a mapping: an unread key in an open mapping leaves that role unplaced
    with no error anywhere, which is the failure this exists to remove.
    """

    trainer: Placement = Placement()
    actor: Placement = Placement()
    inference_server: Placement = Placement()
    params: Placement = Placement()
    metrics: Placement = Placement()
    checkpoints: Placement = Placement()


def reserve(placement: Placement, size: int, strategy: Strategy = "PACK") -> PlacementGroup:
    """Reserve ``size`` members' worth of ``placement`` for a Gang, together or not at all.

    Bundles and their label selectors are derived from the claim the members themselves use, so a
    member fits its bundle by construction. Returns Ray's own type rather than a wrapper, so a
    hand-built group with bundles brrrl does not model passes to ``Placement.apply`` unchanged.
    """
    bundle = placement.bundle()
    if not bundle:
        raise ValueError(f"A reservation needs a claim, and {placement} asks for no resources.")
    labels = [dict(placement.label_selector)] * size if placement.label_selector else None
    return placement_group([bundle] * size, strategy, bundle_label_selector=labels)
