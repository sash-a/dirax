from collections.abc import Sequence

import jax.numpy as jnp
from jax import tree as jtree


def tree_transpose[T](trees: Sequence[T]) -> T:
    """Convert a list of trees of into a tree of lists."""
    return jtree.map(lambda *leaves: jnp.stack(leaves, axis=0), *trees)


def tree_add_leading_dim[T](tree: T) -> T:
    """Add a leading dim to all leaves of a tree."""
    return jtree.map(lambda leaf: leaf[jnp.newaxis], tree)


def tree_swapaxes[T](tree: T, axis1: int, axis2: int) -> T:
    """Swap the leading two axes of every leaf: a Roller's ``(T+1, E, ...)`` to ``(E, T+1, ...)``."""
    return jtree.map(lambda leaf: jnp.swapaxes(leaf, axis1, axis2), tree)
