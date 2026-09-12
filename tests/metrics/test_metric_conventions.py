"""The conventions the metrics layer rests on, worth more than any one processor in it.

Each is an architectural commitment an ADR rests on, so a change that breaks one should fail here
rather than surface as a series logged against the wrong clock, or as an import error on a cluster
that installed neither tracking backend.
"""

import ast
import importlib
import inspect
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

import brrrl.metrics
from brrrl.metrics.logger import Logger, MultiLogger, NamespacedProcessor
from brrrl.metrics.stdout import StdoutLogger

FORBIDDEN_IMPORTS = ("ray", "brrrl.distributed", "brrrl.algorithms")
OPTIONAL_BACKENDS = {"mlflow": "mlflow.py", "wandb": "wandb.py"}
THE_TRAIT = ("log", "log_artifact", "close")
THE_STEP = "step"


def implementations() -> list[type]:
    """Every Logger in the package, with the optional backends included where they are installed."""
    found: list[type] = [StdoutLogger, NamespacedProcessor, MultiLogger]
    for package, backend in (("mlflow", "MLflowLogger"), ("wandb", "WandbLogger")):
        with suppress(ImportError):
            found.append(getattr(importlib.import_module(f"brrrl.metrics.{package}"), backend))
    return found


def test_the_module_reaches_no_ray_and_nothing_from_the_layers_above_it() -> None:
    """ADR-0004: an algorithm composed of these must train in one process with no Ray in its
    transitive imports. A chain is handed to a component, and knows nothing about where it runs."""
    assert [name for name in imported() if name.startswith(FORBIDDEN_IMPORTS)] == []


def test_each_optional_backend_is_reached_from_its_own_module_and_nowhere_else() -> None:
    """mlflow and wandb are extras, so a run that installed neither still needs the accumulator, the
    processors and the collector. One stray import at the top of a shared module breaks every run."""
    for package, module in OPTIONAL_BACKENDS.items():
        elsewhere = [source.name for source in sources() if source.name != module and reaches(source, package)]

        assert elsewhere == [], f"{package} is an extra, so {elsewhere} must not import it"


def test_every_logger_names_the_position_it_is_given_step() -> None:
    """ADR-0005: ``Logger.log`` takes an absolute position, which maps 1:1 onto what a backend
    stores. A stage that named it anything else would be counting, and counting lives in one place."""
    assert positions_of("log") == {THE_STEP}


def test_nothing_that_counts_transitions_names_its_count_step() -> None:
    """The load-bearing half of ADR-0005: ``record`` and ``collect`` share an arity and a type with
    ``log``, so the parameter name is the only thing stopping a recorder from structurally
    satisfying Logger and nesting inside its own chain — a second clock nobody declared."""
    counting = positions_of("record") | positions_of("collect")

    assert counting and THE_STEP not in counting


@pytest.mark.parametrize("implementation", implementations(), ids=lambda cls: cls.__name__)
def test_every_stage_of_a_chain_takes_the_traits_own_signatures(implementation: type) -> None:
    """Nothing inherits here, so a Processor and the backend it wraps only compose because their
    signatures agree. A renamed argument or a dropped default still composes positionally, right up
    until the stage above it passes one by keyword."""
    for method in THE_TRAIT:
        assert arguments(getattr(implementation, method)) == arguments(getattr(Logger, method)), method


def arguments(method: Any) -> list[tuple[str, Any]]:
    return [(name, argument.default) for name, argument in inspect.signature(method).parameters.items()]


def sources() -> list[Path]:
    return sorted(Path(next(iter(brrrl.metrics.__path__))).glob("*.py"))


def imported() -> set[str]:
    return {name for source in sources() for name in imported_names(source)}


def imported_names(source: Path) -> set[str]:
    """Every module and every name the file imports."""
    tree = ast.parse(source.read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]
    names = {alias.name for node in imports for alias in node.names}
    modules = {node.module for node in imports if isinstance(node, ast.ImportFrom) and node.module}
    return names | modules


def reaches(source: Path, package: str) -> bool:
    return any(name == package or name.startswith(f"{package}.") for name in imported_names(source))


def positions_of(method: str) -> set[str]:
    """What every ``method`` in the package calls the argument after its metrics."""
    definitions = [
        node
        for source in sources()
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == method
    ]
    return {node.args.args[2].arg for node in definitions}
