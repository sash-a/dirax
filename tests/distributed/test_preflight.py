"""The preflight against a real scheduler — the one behaviour a pure test cannot see.

A hang is only observable against something that schedules, so these start a small local Ray
and are the only tests here that do. The instance is pinned rather than inherited from the
host, so the same claims are satisfiable on a laptop and in CI.
"""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pytest
import ray
from ray.util.placement_group import remove_placement_group

from brrrl.distributed.placement.component import Behaviour
from brrrl.distributed.placement.placement import Placement, reserve
from brrrl.distributed.placement.preflight import Placed, Unplaceable, preflight

WATCHED = ("JAX_PLATFORMS", "XLA_PYTHON_CLIENT_MEM_FRACTION", "BRRRL_JOB_WIDE")


class Probe:
    """A Component that reports what it was actually given."""

    RAY_OPTIONS: ClassVar[Behaviour] = {}

    def report(self) -> tuple[dict[str, float], dict[str, str]]:
        assigned = ray.get_runtime_context().get_assigned_resources()
        return assigned, {name: os.environ[name] for name in WATCHED if name in os.environ}


@pytest.fixture(scope="module")
def cluster() -> Iterator[None]:
    """``PYTHONPATH`` is what lets a worker import ``Probe`` from this file.

    ``BRRRL_JOB_WIDE`` rides along as an ordinary job-level variable, to be checked against the
    ones brrrl derives per role.
    """
    job = {"PYTHONPATH": str(Path(__file__).parent), "BRRRL_JOB_WIDE": "kept"}
    ray.init(num_cpus=4, num_gpus=0, include_dashboard=False, log_to_driver=False, runtime_env={"env_vars": job})
    yield
    ray.shutdown()


def test_a_satisfiable_claim_clears_the_preflight(cluster: None) -> None:
    claim = Placement(num_cpus=1)
    probe = claim.apply(Probe).remote()

    preflight([Placed("probe", claim, probe)], timeout=60)


def test_an_unsatisfiable_claim_fails_with_the_claim_beside_what_the_cluster_offers(cluster: None) -> None:
    """A misspelled resource name costs seconds rather than a queue slot, and reads without a dashboard."""
    claim = Placement(num_cpus=1, resources={"learnr": 1})
    probe = claim.apply(Probe).remote()

    with pytest.raises(Unplaceable) as failure:
        preflight([Placed("probe", claim, probe)], timeout=2)

    assert "probe" in str(failure.value)
    assert "learnr" in str(failure.value)
    assert "CPU" in str(failure.value)


def test_the_wait_is_narrated_while_a_claim_is_pending(cluster: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Queueing for capacity on a shared cluster is normal, and must look different from a broken run."""
    claim = Placement(resources={"learnr": 1})
    probe = claim.apply(Probe).remote()

    with pytest.raises(Unplaceable):
        preflight([Placed("probe", claim, probe)], timeout=2, interval=0.5)

    narration = capsys.readouterr().err  # tqdm writes its bar to stderr.

    assert "learnr" in narration
    assert "component" in narration  # tqdm's own unit, so the bar carried it and not a Ray warning.


def test_a_reservation_is_confirmed_and_its_bundles_are_the_claim(cluster: None) -> None:
    """Members fit their bundles by construction because both come from the one claim."""
    claim = Placement(num_cpus=1)
    group = reserve(claim, size=2)

    preflight([Placed("gang", claim, group)], timeout=60)
    bundles = group.bundle_specs
    remove_placement_group(group)  # A reservation holds its resources until removed.

    assert bundles == [{"CPU": 1.0}, {"CPU": 1.0}]


def test_a_labelled_claim_places_its_gang_members_inside_the_reservation(cluster: None) -> None:
    """Routing a role by label and reserving it as a Gang have to compose, because the bundle's
    selector and the member's own both come from the one claim."""
    claim = Placement(num_cpus=1, label_selector={"ray.io/node-id": ray.nodes()[0]["Labels"]["ray.io/node-id"]})
    group = reserve(claim, size=2)
    member = claim.apply(Probe, group, index=1).remote()

    preflight([Placed("gang", claim, group), Placed("member", claim, member)], timeout=60)
    assigned, _ = ray.get(member.report.remote())
    remove_placement_group(group)

    assert assigned == {"CPU": 1.0}


def test_a_placed_component_gets_the_resources_and_environment_its_claim_implies(cluster: None) -> None:
    """The algebra reaching a real process. Ray leaves ``CUDA_VISIBLE_DEVICES`` unset for this
    actor, so the derived platform is the only thing keeping it off a card it never claimed."""
    claim = Placement(num_cpus=1)
    probe = claim.apply(Probe).remote()

    assigned, environment = ray.get(probe.report.remote())

    assert assigned == {"CPU": 1.0}
    assert environment["JAX_PLATFORMS"] == "cpu"


def test_job_wide_environment_variables_survive_the_ones_brrrl_derives(cluster: None) -> None:
    """Ray merges job-level and per-actor ``env_vars`` per key, so setting a per-role variable
    must not silently drop what the caller set on their own ``ray.init``."""
    probe = Placement(num_cpus=1).apply(Probe).remote()

    _, environment = ray.get(probe.report.remote())

    assert environment["BRRRL_JOB_WIDE"] == "kept"
    assert environment["JAX_PLATFORMS"] == "cpu"
