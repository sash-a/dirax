"""The conventions every primitive inherits, worth more than any one module.

Each is an architectural commitment an ADR rests on, so a change that breaks one should
fail here rather than surface as a puzzling runtime error three layers away.
"""

import ast
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

import brrrl.networks
from brrrl.networks.distribution import TanhNormal
from brrrl.networks.head import ContinuousHead, DiscreteHead, MaskedDiscreteHead, ValueHead
from brrrl.networks.torso import ImpalaCNNTorso, MLPTorso, NatureCNNTorso

NUM_LANES = 4
EMBEDDING = 8
NUM_ACTIONS = 4
PIXELS = 16
CHANNELS = 3

FORBIDDEN_IMPORTS = ("brrrl.acting", "brrrl.distributed", "brrrl.algorithms", "ray", "echo")
FORBIDDEN_TYPES = {"Transition", "Trajectory"}


def logits_of(output: distrax.Categorical) -> jax.Array:
    return jnp.asarray(output.logits)


def loc_of(output: TanhNormal) -> jax.Array:
    return output.loc


def itself(output: jax.Array) -> jax.Array:
    return output


def embedding() -> tuple[jax.Array, ...]:
    return (jnp.ones((NUM_LANES, EMBEDDING)),)


def embedding_and_mask() -> tuple[jax.Array, ...]:
    return (jnp.ones((NUM_LANES, EMBEDDING)), jnp.ones((NUM_LANES, NUM_ACTIONS), bool))


def frame() -> tuple[jax.Array, ...]:
    return (jnp.ones((NUM_LANES, PIXELS, PIXELS, CHANNELS)),)


class Primitive(NamedTuple):
    """One module under test, with what to apply it to and where a dtype is readable off it."""

    build: Callable[..., nn.Module]
    inputs: Callable[[], tuple[jax.Array, ...]]
    array_of: Callable[[Any], jax.Array]


PRIMITIVES = [
    pytest.param(Primitive(lambda **kw: MLPTorso(layer_sizes=(6,), **kw), embedding, itself), id="mlp"),
    pytest.param(
        Primitive(lambda **kw: DiscreteHead(num_actions=NUM_ACTIONS, **kw), embedding, logits_of), id="discrete"
    ),
    pytest.param(
        Primitive(lambda **kw: MaskedDiscreteHead(num_actions=NUM_ACTIONS, **kw), embedding_and_mask, logits_of),
        id="masked",
    ),
    pytest.param(
        Primitive(lambda **kw: ContinuousHead(num_actions=NUM_ACTIONS, **kw), embedding, loc_of), id="continuous"
    ),
    pytest.param(Primitive(lambda **kw: ValueHead(**kw), embedding, itself), id="value"),
    pytest.param(Primitive(NatureCNNTorso, frame, itself), id="nature"),
    pytest.param(Primitive(ImpalaCNNTorso, frame, itself), id="impala"),
]


@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_computation_dtype_reaches_the_output(primitive: Primitive, apply_module: Callable[..., Any]) -> None:
    output = apply_module(primitive.build(dtype=jnp.bfloat16), *primitive.inputs())

    assert primitive.array_of(output).dtype == jnp.bfloat16


@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_params_stay_in_their_own_dtype_while_computation_moves(
    primitive: Primitive, apply_module: Callable[..., Any]
) -> None:
    """The mixed precision an inference server wants: master weights kept, forward pass cheap.

    A half-precision ``param_dtype`` is not on offer — orthogonal initialisation runs a QR and
    there is no half-precision QR kernel — so single-precision storage is the thing to pin.
    """
    module = primitive.build(param_dtype=jnp.float32, dtype=jnp.bfloat16)
    params = module.init(jax.random.key(0), *primitive.inputs())

    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))
    assert primitive.array_of(apply_module(module, *primitive.inputs())).dtype == jnp.bfloat16


def test_params_are_a_plain_nested_mapping_of_arrays() -> None:
    """ADR-0002 and ADR-0006: nested dicts of arrays flatten identically under JAX and optree.

    An NNX state or a registered dataclass would fail both asserts, and is one opaque leaf to
    the optree registry echo flattens with.
    """
    params = MLPTorso(layer_sizes=(6,)).init(jax.random.key(0), jnp.ones((NUM_LANES, EMBEDDING)))

    assert isinstance(params, dict)
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(params))


def test_the_package_imports_nothing_from_the_acting_or_placement_layers() -> None:
    """ADR-0004: an algorithm's standalone entry point must reach none of it.

    Nor any observation container: primitives take arrays, so an algorithm changing its
    observation or Trajectory type touches only that algorithm.
    """
    sources = Path(next(iter(brrrl.networks.__path__))).glob("*.py")
    imported = {name for source in sources for name in imported_names(source)}

    assert not [name for name in imported if name.startswith(FORBIDDEN_IMPORTS)]
    assert not imported & FORBIDDEN_TYPES


def imported_names(source: Path) -> set[str]:
    """Every module and every name the file imports."""
    tree = ast.parse(source.read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]
    names = {alias.name for node in imports for alias in node.names}
    modules = {node.module for node in imports if isinstance(node, ast.ImportFrom) and node.module}
    return names | modules
