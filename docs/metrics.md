# Metrics

Every metric in a run is logged against **one** x-axis, the Step: cumulative
transitions produced by Actors. Learner `Version`, transitions consumed, wall-clock —
those are scalars you log, never axes. One axis is what lets a return curve and a loss
curve overlay; see [ADR 0005](adr/0005-one-metrics-axis.md) for why.

Metrics only flow one way. Many components record, one collector stamps the Step, one
chain writes:

```
record  →  collect  →  log
```

## Two protocols

| | Who implements it | Who calls it |
|---|---|---|
| `MetricRecorder.record(metrics, transitions=0)` | `LocalMetricRecorder`, `RayMetricRecorder` | components |
| `Logger.log(metrics, step)` + `close()` | processors and backends | the collector |

A component holds a `MetricRecorder` and never learns the Step. It reports how far it
moved the world — `transitions` — and only Actors move it, so everyone else omits the
argument. The collector accumulates those increments into the Step and hands an
absolute position to a `Logger`.

Namespace, and whether an origin is windowed or logged raw, are fixed per component
and bound at wiring time rather than passed per call.

## The collector

`MetricCollector` does two jobs: it owns the Step clock, and it routes each `Origin`
to a logger chain. A collector is centralised, there is only 1. But there may be many reporters.
A reporter tells the collector where the logs come from and how many steps these logs represent.
The collector is then able to increment the central step counter and route to the correct logger.

## Processors

`MetricCollector` does two jobs: it owns the Step clock, and it routes each `Origin`
to a logger chain. There is one collector and many recorders. A recorder tells the
collector which `Origin` the metrics came from and how many transitions they account
for; the collector increments the Step by that count and routes to the right chain.

## Wiring

Each `Origin` routes to a chain of processors over one shared backend. Three entries,
one MLflow run:

```python
def build_collector() -> MetricCollector:
    backend = MLflowLogger(...)
    return MetricCollector({
        Origin.ACTOR:   WindowedProcessor(NamespacedProcessor("actor", backend), every=10_000),
        Origin.LEARNER: NamespacedProcessor("learner", backend),
        Origin.EVAL:    WindowedProcessor(NamespacedProcessor("eval", backend), every=10_000),
    })

collector = RayMetricCollector.remote(build_collector)
actor = Actor(..., metrics=RayMetricRecorder(collector, Origin.ACTOR))
trainer = Trainer(..., metrics=RayMetricRecorder(collector, Origin.LEARNER))
```

Actors are windowed because many of them record often; the Trainer is raw because one
of it records rarely, and averaging a loss over 10k steps would hide the spikes.

`RayMetricCollector` is single-threaded — Ray serialises calls into it, so the backend
never sees concurrent writers and needs no lock. `RayMetricRecorder` is
fire-and-forget, so nothing on the hot path blocks on a slow backend.

## Recording

```python
# Actor: advances the clock.
self.metrics.record({"return": [1.0, 3.0]}, transitions=512)

# Trainer: no transitions; version rides along as a scalar.
self.metrics.record({**metrics, "version": [version], "consumed": [consumed]})
```

Which lands as:

```
  512  actor/return/mean = 2.0
  512  learner/loss = 0.5, learner/version = 1, learner/consumed = 1024
```

Nested metrics get flattened first — `flatten` joins keys with `/` and raises on a
collision rather than silently dropping one:

```python
flatten({"episode": {"return": [1.0, 3.0]}})   # {"episode/return": [1.0, 3.0]}
```

## Shutdown

`close()` is on `Logger`, not `MetricRecorder`: whoever builds a chain owns it, and
components are handed a recorder they don't own. Skip it and you lose the last partial
window.

```python
ray.get(collector.close.remote())
```

## Without Ray

`main()` builds the same collector and takes recorders off it directly, so it gets the
same routing and the same single clock
([ADR 0004](adr/0004-algorithms-run-without-ray.md)):

```python
collector = build_collector()
actor_metrics = LocalMetricRecorder(collector, Origin.ACTOR)
learner_metrics = LocalMetricRecorder(collector, Origin.LEARNER)
```

One collector, one clock. Give the Actor and the Trainer a collector each and the
Trainer's clock sits at zero forever.

With one Roller there are no concurrent producers to aggregate across, so the window
is optional here — drop `WindowedProcessor` from the chain if you want the raw rate.
