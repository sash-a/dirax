# Linen, not NNX

Networks are Flax **Linen** modules, applied as `apply(params, ...)`. Flax's own docs
recommend NNX for new projects and put Linen in maintenance mode, so this needs a
reason beyond familiarity.

NNX's whole value is the stateful module — params living inside the object, mutated in
place. dirax cannot use that style anywhere. `LearnerState` carries params through a
scanned pure function, ADR-0003 forecasts a jitted Roller family built on explicit
params, and every param crossing to a Param sink or a checkpoint is a plain tree of
arrays. Adopting NNX means calling `nnx.split` at every boundary and never once holding
a live module: paying its costs and collecting none of its benefits.

## Consequences

`Params` stays `flax.typing.VariableDict` — a nested dict of arrays, which is the
cheapest possible thing to publish, checkpoint and shard.
