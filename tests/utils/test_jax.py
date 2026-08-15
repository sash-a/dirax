"""What the tree helpers must hold: the structure survives and only the shapes move.

Each is one ``jax.tree.map``, so what is worth asserting is the contract around the numpy call
rather than the call itself. Two things carry the file. Structure preservation is load-bearing —
PPO swaps a Trajectory's axes and puts the result straight back with ``_replace``, which accepts
only the NamedTuple it took out. And leaves of unequal rank are the normal case, not an edge one:
an observation carries trailing axes a reward does not, which is why every helper is named for a
leading axis and must leave the rest alone.
"""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytest
from jax import tree as jtree

from brrrl.utils.jax import tree_add_leading_dim, tree_swapaxes, tree_transpose

SLOTS, ENVS, OBS = 5, 3, 2


class Slots(NamedTuple):
    """A Trajectory in miniature: a nested observation with a trailing axis, a flat reward beside it."""

    observation: dict[str, np.ndarray]
    reward: np.ndarray


@pytest.fixture
def slots() -> Slots:
    """``(T, E, ...)`` leaves, every element distinct, so a misplaced axis reads as a wrong value."""
    generator = np.random.default_rng(0)
    return Slots(
        observation={"pixels": generator.normal(size=(SLOTS, ENVS, OBS))},
        reward=generator.normal(size=(SLOTS, ENVS)),
    )


def shifted(tree: Slots, by: float) -> Slots:
    """The same tree a known distance away, so which one a stacked row came from is readable."""
    return jtree.map(lambda leaf: leaf + by, tree)


@pytest.mark.parametrize(
    "helper",
    [
        pytest.param(lambda tree: tree_swapaxes(tree, 0, 1), id="tree_swapaxes"),
        pytest.param(tree_add_leading_dim, id="tree_add_leading_dim"),
        pytest.param(lambda tree: tree_transpose([tree, tree]), id="tree_transpose"),
    ],
)
def test_the_tree_comes_back_the_type_it_went_in(helper: Callable[[Slots], Slots], slots: Slots) -> None:
    """PPO feeds the result straight into ``_replace``, which a bare dict or list would fail."""
    mapped = helper(slots)

    assert isinstance(mapped, Slots)
    assert jtree.structure(mapped) == jtree.structure(slots)


def test_swapaxes_moves_the_named_axes_of_every_leaf(slots: Slots) -> None:
    """One call over leaves of unequal rank: the trailing axes ride along untouched."""
    swapped = tree_swapaxes(slots, 0, 1)

    assert swapped.observation["pixels"].shape == (ENVS, SLOTS, OBS)
    assert swapped.reward.shape == (ENVS, SLOTS)


def test_swapaxes_transposes_rather_than_reshapes(slots: Slots) -> None:
    """The one way this goes wrong quietly: a reshape gives the right shape and the wrong values."""
    swapped = tree_swapaxes(slots, 0, 1)

    assert np.allclose(swapped.reward, slots.reward.T)
    assert np.allclose(swapped.observation["pixels"][1, 2], slots.observation["pixels"][2, 1])


def test_swapaxes_twice_is_the_identity(slots: Slots) -> None:
    there_and_back = tree_swapaxes(tree_swapaxes(slots, 0, 1), 0, 1)

    assert np.allclose(there_and_back.reward, slots.reward)
    assert np.allclose(there_and_back.observation["pixels"], slots.observation["pixels"])


def test_swapaxes_takes_axes_other_than_the_leading_pair() -> None:
    """The axes are arguments; the leading pair is what the docstring names as the use, not the limit."""
    tree = {"pixels": np.arange(SLOTS * ENVS * OBS).reshape(SLOTS, ENVS, OBS)}

    swapped = tree_swapaxes(tree, 1, 2)

    assert swapped["pixels"].shape == (SLOTS, OBS, ENVS)
    assert np.allclose(swapped["pixels"][0, 1], tree["pixels"][0, :, 1])


def test_transpose_stacks_the_trees_under_one_leading_axis(slots: Slots) -> None:
    stacked = tree_transpose([slots, slots, slots])

    assert stacked.observation["pixels"].shape == (3, SLOTS, ENVS, OBS)
    assert stacked.reward.shape == (3, SLOTS, ENVS)


def test_transpose_keeps_each_tree_at_its_own_index(slots: Slots) -> None:
    """A list of trees becomes a tree of lists, so row ``i`` has to still be the ``i``th tree."""
    trees = [shifted(slots, step) for step in range(3)]

    stacked = tree_transpose(trees)

    for step, tree in enumerate(trees):
        assert np.allclose(stacked.reward[step], tree.reward)
        assert np.allclose(stacked.observation["pixels"][step], tree.observation["pixels"])


def test_transpose_of_a_single_tree_is_a_leading_dim(slots: Slots) -> None:
    """The two helpers meet at one tree: stacking it is the same as making room for it."""
    stacked = tree_transpose([slots])
    added = tree_add_leading_dim(slots)

    assert stacked.reward.shape == added.reward.shape
    assert np.allclose(stacked.reward, added.reward)


def test_add_leading_dim_makes_room_for_one(slots: Slots) -> None:
    added = tree_add_leading_dim(slots)

    assert added.observation["pixels"].shape == (1, SLOTS, ENVS, OBS)
    assert added.reward.shape == (1, SLOTS, ENVS)


def test_add_leading_dim_moves_no_value(slots: Slots) -> None:
    added = tree_add_leading_dim(slots)

    assert np.allclose(added.reward[0], slots.reward)
    assert np.allclose(added.observation["pixels"][0], slots.observation["pixels"])
