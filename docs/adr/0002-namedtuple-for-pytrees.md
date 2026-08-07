# NamedTuple for every pytree container

Every container that is a pytree — Trajectory, Transition, learner state,
policy output — is a `NamedTuple`, never a dataclass. This is forced by having
two pytree registries in play: JAX has its own, and echo flattens with optree.

A `@jax.tree_util.register_dataclass` dataclass is **one opaque leaf** to optree,
and registering it with optree doesn't help — optree rejects the default
namespace, which is the one echo uses:

```
jax-registered dataclass   jax leaves=2   optree leaves=1
NamedTuple                 jax leaves=2   optree leaves=2
optree.register_pytree_node(..., namespace='')  →  ValueError
```

`NamedTuple` works natively in both registries with no registration at all, so
there is nothing to forget when adding an algorithm.

## Consequences

`class PPOTrajectory(NamedTuple)` reads as inheritance against our
final-classes-only rule. It is a declaration form, not behavioural inheritance —
nothing is inherited and it cannot be meaningfully subclassed.

NamedTuples are tuples, so `obs, action = trajectory` silently works and
iterates in field order. Accept it or lint it.
