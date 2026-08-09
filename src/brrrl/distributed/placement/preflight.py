import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import ray
from ray import ObjectRef
from ray.actor import ActorProxy
from ray.util.placement_group import PlacementGroup

from brrrl.distributed.placement.placement import Placement


class Unplaceable(RuntimeError):
    """A claim the cluster never satisfied."""


@dataclass(frozen=True, slots=True)
class Placed:
    """One thing the preflight waits on: what to call it, what it claimed, and what came back."""

    role: str
    placement: Placement
    handle: ActorProxy[Any] | PlacementGroup

    def readiness(self) -> ObjectRef:
        """A reference that resolves once this is scheduled, and stays pending while it is not."""
        if isinstance(self.handle, PlacementGroup):
            return self.handle.ready()
        return self.handle.__ray_ready__.remote()


def announce(message: str) -> None:
    print(message, flush=True)


def preflight(
    placed: Sequence[Placed],
    timeout: float = 600,
    interval: float = 10,
    narrate: Callable[[str], None] = announce,
) -> None:
    """Block until every claim in ``placed`` is satisfied, or fail naming the ones that were not.

    Without it an unschedulable Component sits pending forever, and a run short four of sixty
    Actors trains to completion at reduced throughput with nothing reported anywhere.

    Args:
        placed: Every handle and reservation the run depends on, with the claim each came from.
        timeout: Seconds to allow before giving up. Queueing for capacity is normal, so this is
            generous and ``narrate`` is what makes a slow start legible while it happens.
        interval: Seconds between narrated lines while anything is still pending.
        narrate: Where progress goes. Takes a line, so a caller holding a progress bar passes
            its update instead.

    Raises:
        Unplaceable: A claim was still unsatisfied at the deadline.
    """
    pending = {item.readiness(): item for item in placed}
    deadline = time.monotonic() + timeout
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise Unplaceable(diagnosis(list(pending.values()), timeout))
        ready, _ = ray.wait(list(pending), num_returns=len(pending), timeout=min(interval, remaining))
        ray.get(ready)  # A Component that died building itself is a real error, not a slow one.
        for reference in ready:
            del pending[reference]
        if pending:
            narrate(f"Waiting on {claims(list(pending.values()))} — {deadline - time.monotonic():.0f}s left")


def claims(pending: Sequence[Placed]) -> str:
    """The unsatisfied claims, one entry per distinct claim rather than one per process."""
    counts = Counter(f"{item.role} claiming {item.placement}" for item in pending)
    return "; ".join(f"{count}x {claim}" for claim, count in counts.items())


def diagnosis(pending: Sequence[Placed], timeout: float) -> str:
    """What was asked for, beside what the cluster actually offers."""
    nodes = "\n".join(
        f"  {node['NodeManagerAddress']}: resources={node['Resources']} labels={node['Labels']}"
        for node in ray.nodes()
        if node["Alive"]
    )
    return f"Unsatisfied after {timeout:g}s: {claims(pending)}\nThe cluster offers:\n{nodes}"
