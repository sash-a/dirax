# One metrics axis; every other clock is a metric

The step every metric is logged against is cumulative transitions **produced** by
Actors. Nothing else is ever a step. The Trainer's `Version`, and its count of
transitions **consumed**, are logged as scalars inside `Metrics`.

Both counts are exact and they differ: in an async run, produced exceeds consumed
by the in-flight queue plus whatever was dropped for staleness. Giving each its
own axis offsets the return curve from the loss curve while leaving them looking
comparable, so a loss at step 1M would be read against a return from a different
moment. Stamping learner metrics with the current produced count instead reads as
"the loss measured once this much environment interaction had happened", which is
what overlaying the two curves is supposed to mean.

Produced steps is also the only axis comparable across topologies. Update count
is a function of batch size, and stale data that was dropped was still paid for.

A backend can re-key a series on any scalar it was given, but two series logged
against different steps can never be joined. The axis is the one choice that is
hard to reverse; keeping it single keeps everything else recoverable.

## One log method; per-source behaviour is bound at construction

`MetricsSink.log(metrics, num_transitions=0)` is the whole component-facing surface.
Only Actors pass a nonzero count — the Trainer, evaluators and the checkpointer
produce zero transitions, which is true rather than a sentinel.

Namespace, and whether a source is windowed or logged raw, are properties of the
*caller*, not of the call: they are fixed for a component and known at wiring
time. So they are bound into the sink it is handed (`Source`) and resolved by a
routing table in `MetricsStore`, rather than appearing in the signature.

`num_transitions` must not also select that behaviour. Evaluators fold — several of
them aggregate per key — while advancing the clock not at all, so "advances the
clock" and "is aggregated" are separate axes and no heuristic on `num_transitions`
can express both.

The cost is that reading `MetricsSink` no longer tells you which sources exist or
how each is treated; the `Source` enum and the routing table do. In exchange
`num_transitions` means exactly one thing, a component cannot tag itself wrong, and a
new source is a new row rather than a protocol change.

## Counting, windowing and shutdown

Counting lives in `MetricCountingSink`, never in a backend. `Logger.log(metrics,
step)` takes an absolute position and maps 1:1 onto `mlflow.log_metrics` and
`wandb.log`, so the sink is the single place the unit changes from increment to
position. The differing parameter names are load-bearing: the two protocols share
an arity and a type, so `step` vs `num_transitions` is what stops a sink from
structurally satisfying `Logger` and nesting inside itself.

Windowing is a `Logger` decorator sitting *below* the counting sink, not a
`MetricsSink` wrapping it. Above the sink it would have to count transitions
itself to find its boundaries, giving a second clock; below it, the step is
already resolved and a boundary is just arithmetic on it.

`close()` is on `Logger` and not on `MetricsSink`, because whoever builds a chain
owns its lifetime and components are handed a sink they do not own. It must be
idempotent: source chains share one backend, so it is reached once per chain.
Without it the last partial window is lost, and `RayMetricsSink` is
fire-and-forget so nothing else would flush it.

`MetricsStore` takes no `concurrency_groups`, so Ray serialises every call into
it. That is what removes the per-source locks and lets a non-thread-safe backend
be called without one. Adding concurrency groups for throughput means adding that
lock back.

`MetricAccumulator` reduces to mean/std/min/max only. A monotonic counter through
it comes out as the window's midpoint, so `version` and `consumed` need a sum or
last-value reducer before they can ride inside windowed `Metrics`.
