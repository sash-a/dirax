"""The conventions the acting layer rests on, worth more than any one type in it.

Each is an architectural commitment an ADR rests on, so a change that breaks one should fail here
rather than surface as a puzzling runtime error three layers away.
"""

import ast
from pathlib import Path

import brrrl.acting

FORBIDDEN_IMPORTS = ("ray", "brrrl.distributed", "brrrl.algorithms")
ECHO_TRANSPORT = {"Server", "Sample", "SampleInfo", "HistSnapshot", "TcpClient", "TcpTransport"}
THE_LOOP = "roller.py"
THE_SEAM = "env_stepper.py"


def test_the_loop_reaches_no_jax() -> None:
    """The autoreset discipline is the delicate part, and it should read without anyone having to
    reason about device placement. Keys live in the Policy and the Version stamp is a numpy int."""
    assert [name for name in imported_names(module(THE_LOOP)) if name.startswith("jax")] == []


def test_the_module_reaches_no_ray_and_nothing_from_the_layers_above_it() -> None:
    """ADR-0004: an algorithm composed of these must train in one process with no Ray in its
    transitive imports, and a Roller knows nothing about where anything else lives."""
    assert [name for name in imported() if name.startswith(FORBIDDEN_IMPORTS)] == []


def test_the_module_takes_echos_accumulator_and_none_of_its_transport() -> None:
    """ADR-0004 bars the transport, not the buffer: the accumulator is arithmetic on numpy arrays,
    while a client or a server is the distribution that must stay wrapped around an algorithm."""
    assert imported() & ECHO_TRANSPORT == set()
    assert "TrajectoryAccumulator" in imported()


def test_gymnasium_reaches_no_further_than_the_env_stepper() -> None:
    """The stepper is the seam that keeps gymnasium out of every signature above it, which is what
    leaves room for a stepper this repo builds itself."""
    elsewhere = {source.name for source in sources() if source.name != THE_SEAM}

    assert [name for name in elsewhere if "gymnasium" in imported_names(module(name))] == []


def sources() -> list[Path]:
    return sorted(Path(next(iter(brrrl.acting.__path__))).glob("*.py"))


def module(name: str) -> Path:
    return next(source for source in sources() if source.name == name)


def imported() -> set[str]:
    return {name for source in sources() for name in imported_names(source)}


def imported_names(source: Path) -> set[str]:
    """Every module and every name the file imports."""
    tree = ast.parse(source.read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]
    names = {alias.name for node in imports for alias in node.names}
    modules = {node.module for node in imports if isinstance(node, ast.ImportFrom) and node.module}
    return names | modules
