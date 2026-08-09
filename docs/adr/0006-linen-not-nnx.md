# Linen, not NNX

Networks are Flax **Linen** modules, applied as `apply(params, ...)`. Flax's own docs
recommend NNX for new projects and put Linen in maintenance mode, so this needs a
reason beyond familiarity.

NNX's whole value is the stateful module — params living inside the object, mutated in
place. brrrl cannot use that style anywhere. `LearnerState` carries params through a
scanned pure function, ADR-0003 forecasts a jitted Roller family built on explicit
params, and every param crossing to a Param sink or a checkpoint is a plain tree of
arrays. Adopting NNX means calling `nnx.split` at every boundary and never once holding
a live module: paying its costs and collecting none of its benefits.

The measurements, on a small MLP at batch 64:

```
linen  jit(apply)(params, x)         15.7 us/call
nnx    jit(merge + call)(state, x)   16.8 us/call     functional style, +7%
nnx    nnx.jit(module, x)           154.3 us/call     idiomatic style, ~10x
```

The one style worth adopting NNX for is the one that costs ten times Linen's dispatch —
which an inference server cannot pay. `nnx.State` is also a single opaque leaf to
optree, the exact failure ADR-0002 exists to avoid, so params would have to be converted
at every boundary anyway.

## Consequences

`Params` stays `flax.typing.VariableDict` — a nested dict of arrays, which is the
cheapest possible thing to publish, checkpoint and shard.

Every JAX RL network in the wild is Linen, so a network from Stoix, Mava, PureJaxRL or
Brax pastes in unmodified. That is most of what "hackable" means here.

We are starting a new library on an API in maintenance mode, and that is a real bet
rather than a free choice. What limits the damage is that ADR-0003 already erases the
network from every protocol: a user with an NNX network supplies an act function built
from `nnx.split`/`nnx.merge` in three lines. Nobody is locked out, and reversing this
changes only our own modules.
