# Parameters reach a Policy down a chain of identical Sources, pushed

`ParamSource` is one method — `get(block: bool) -> VersionedParams | None`,
returning None only when non-blocking and nothing new has landed. `ParamSink` is
one method — `publish(params) -> Version`. Because a Mirror and the Store share
that API, the links compose: **Store → node Mirror → process holder → Policy**,
each one a Source. Whether a Mirror exists at all is a Topology choice rather than
a code change, worth making when Actor fan-out would otherwise overwhelm the Store.

The reader's identity travels in the client half as a routing tag, never in the
signature — CONTEXT.md's Source already says a Source "carries a reference and at
most a routing tag, never the state". That is what keeps the protocol identical for
a local Store and a remote Mirror, and it is the whole reason the chain composes.
`@overload` on `Literal[True]`/`Literal[False]` gives exact types at both call
sites, which is what lets a Policy hold a non-optional `params` rather than an
impossible state it has to check for.

A Mirror `ray.put`s and holds the ref, so parameters exist once per node and every
Actor there reads them zero-copy. Newest overwrites oldest, so a Mirror that fell
behind serves the latest rather than working through a backlog. It and the Actors it
serves are a Gang — one bundle per node under a strict-spread reservation.

## Publishing pushes; reading never blocks

The Sink writes the Store and then every Mirror, fire-and-forget, resting on Ray
executing one handle's calls in order so a Mirror cannot apply v5 then v4. That is
the same bet ADR-0011 already makes for `CheckpointStore`, and the same reason
`max_task_retries` sits outside `Behaviour`. The Sink `ray.put`s once and passes the
ref, so the payload serialises once rather than per Mirror. Mirror handles come from
the Topology at build time, so there is no runtime registration protocol and no
membership to maintain. A Mirror keeps its pull path for cold start and for a
restart that came back empty — a restart then costs a round trip rather than a
node's worth of idle Actors.

Polling would make staleness a function of an interval nobody chose. Instead a
reader holds a **prefetched long-poll**: the blocking sits on the source side inside
a request that is already in flight, so the reader's own check is a zero-timeout
wait and it re-arms immediately.

```python
self.request = [source.get.remote(self.id_ref)]  # armed up front


def get(self, block: bool = False):
    ready, _ = ray.wait(self.request, num_returns=1, timeout=None if block else 0)
    if ready:
        self.request = [source.get.remote(self.id_ref)]  # re-arm at once
        params_ref, version = ray.get(ready[0])
        return VersionedParams(ray.get(params_ref), version)
    return None
```

Making that long-poll edge-triggered needs a one-slot queue per reader on the
Mirror, refilled on publish and with an overflow ignored. Every parked long-poll
holds a concurrency slot, so a Mirror must declare `max_concurrency` of at least its
cohort size plus one for the publish. That is the case ADR-0011's Consequences
predicted — *"`InferenceServer`'s `max_concurrency` plausibly scales with the size of
the cohort it serves, and would be the first case that is not [a constant] — left to
surface rather than designed for"* — and for a Mirror it is a correctness
requirement, not tuning. The reader identity is the **Actor process**, not the
Policy: the Rollers on one Actor read a process-local holder, so a Mirror's
concurrency scales with Actors per node rather than Rollers.

## The write side owns the Version

`publish` assigns the Version and reports it, so a run has one monotonic authority
for it. Nothing checkpoints the Sink, so the driver seeds it on resume: the Trainer
files checkpoints under the Version — natural, since it saves straight after an
update, and it cannot file under the metrics Step because the Metric collector owns
that clock — which makes the Checkpointer's `latest` the last published Version.
Staleness therefore stays comparable across a restart.

The driver also runs the Learner's init and seeds the Store, so nothing stalls at
start-up waiting for a first publish. That is what lets a Sync point never block: a
Policy pulls once when it is built and from then on only adopts what has already
landed.

## Consequences

The staleness bound is deliberately not enforced here: nothing stops an Actor
running ahead of the Trainer. Two mechanisms remain available if it proves
necessary — echo's in-flight send limit on the data path, and a blocking Sync point
— and because the Version rides in the Trajectory the staleness distribution is
measurable before either is chosen.

Each Roller owns its Policy, so a Sync point cannot change the parameters another
Roller is mid-Trajectory on. Sharing one Policy across Rollers would let a
Trajectory span two Versions, and the lock that fixes that would serialise the very
threads it exists to overlap.

A placed Store or Mirror **holds** the in-process `InMemoryParamStore` rather than
reimplementing publish and read, per CONTEXT.md's Store entry. The Ray halves — the
placed Mirror, its client and the fan-out Sink — are decided here and land with the
Actor's run loop; only the in-process Store and the protocol they compose against
are built.
