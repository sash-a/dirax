"""The contract every Component satisfies, worth more than any one component.

Each is an architectural commitment ADR-0011 rests on. A Component that never decided which
Ray options its correctness needs, or one still carrying a decorator that makes it
impossible to instantiate, should fail here rather than surface as a hang on a cluster.
"""

import ast
from pathlib import Path
from typing import Any

import brrrl.distributed.components
from brrrl.acting.policy import Policy
from brrrl.distributed.components.actor import Actor
from brrrl.distributed.components.checkpointer import CheckpointStore
from brrrl.distributed.components.inference_server import InferenceServer, InferenceServerPolicy
from brrrl.distributed.components.metrics import RayMetricCollector
from brrrl.distributed.components.params import ParamStore, RayParamSink, RayParamSource
from brrrl.distributed.components.trainer import Trainer
from brrrl.distributed.placement.component import Behaviour, Component
from brrrl.params.params import ParamSink, ParamSource

COMPONENTS: tuple[type[Component], ...] = (
    Actor,
    CheckpointStore,
    InferenceServer,
    ParamStore,
    RayMetricCollector,
    Trainer,
)
"""Every class in the package Ray places. Typed, so `ty` rejects one that declares no behaviour."""


def views(
    source: RayParamSource, sink: RayParamSink, policy: InferenceServerPolicy[Any, Any]
) -> tuple[ParamSource, ParamSink, Policy[Any, Any]]:
    """Never called: the return type is what makes `ty` hold each client half to the protocol it
    is a view of.

    Nothing in one process ever assigns one of these to its protocol, so a signature that drifted
    from it would otherwise surface as a failure on a cluster.
    """
    return source, sink, policy


def test_every_class_is_either_a_component_or_a_view_of_one() -> None:
    """The package holds placed Components and the handle-holding halves their clients use, and nothing else.

    A new class that is neither has skipped the decision this contract exists to force: an empty
    ``RAY_OPTIONS`` is someone deciding Ray's defaults are right, a missing one is nobody deciding.
    """
    undeclared = [cls.name for cls in classes() if behaviour_of(cls) is None and not holds_a_handle(cls)]

    assert undeclared == []


def test_the_components_are_exactly_the_classes_that_declare_behaviour() -> None:
    """Keeps the statically checked tuple above honest as components are added."""
    declaring = {cls.name for cls in classes() if behaviour_of(cls) is not None}

    assert declaring == {component.__name__ for component in COMPONENTS}


def test_no_component_is_ray_decorated() -> None:
    """A decorated class cannot be instantiated at all, so undecorating them is what makes them unit-testable."""
    decorated = [cls.name for cls in classes() if any("ray.remote" in ast.unparse(node) for node in cls.decorator_list)]

    assert decorated == []


def test_no_declared_behaviour_falls_outside_the_closed_set() -> None:
    """The excluded options are the ones a Component cannot make load-bearing without breaking silently."""
    declared = {key for cls in classes() for key in keys_of(behaviour_of(cls))}

    assert declared <= set(Behaviour.__annotations__)


def classes() -> list[ast.ClassDef]:
    sources = Path(next(iter(brrrl.distributed.components.__path__))).glob("*.py")
    return [node for source in sources for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef)]


def behaviour_of(cls: ast.ClassDef) -> ast.expr | None:
    """What the class assigns to ``RAY_OPTIONS``, or None where it declares nothing."""
    assignments = [node for node in cls.body if isinstance(node, ast.AnnAssign)]
    declarations = [node.value for node in assignments if ast.unparse(node.target) == "RAY_OPTIONS"]
    return next(iter(declarations), None)


def keys_of(behaviour: ast.expr | None) -> list[str]:
    return [ast.literal_eval(key) for key in behaviour.keys if key] if isinstance(behaviour, ast.Dict) else []


def holds_a_handle(cls: ast.ClassDef) -> bool:
    """A client half: it carries a reference to a Component rather than being one."""
    return "ActorHandle" in {ast.unparse(node.annotation) for node in cls.body if isinstance(node, ast.AnnAssign)}
