# NamedTuple for every pytree container

Every container that is a pytree — Trajectory, Transition, State, etc
is a `NamedTuple`, never a dataclass. This is forced by having
two pytree registries in play: JAX has its own, and echo flattens with optree.

`NamedTuple` works natively in both registries with no registration at all, so
there is nothing to forget when adding an algorithm.
