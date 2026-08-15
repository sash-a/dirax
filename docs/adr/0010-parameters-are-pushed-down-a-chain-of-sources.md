# Parameters reach a Policy down a chain of identical Sources, pushed

A Store owns the live parameters. Its write half is a Sink — one method,
`publish(params) -> Version` — and its read half a Source — one method,
`get(block: bool) -> VersionedParams | None`, returning None only when non-blocking and
nothing new has landed. Neither half carries the state, only a reference and at most a
routing tag, so a local Store and one across the cluster are interchangeable to the
caller. A Mirror is a Source holding a copy of the Store's state, and because it presents
that same one method the links compose: **Store → node Mirror → process holder →
Policy**, each one a Source to the link above it.

A Mirror exists so the Actors on a node share one round trip rather than each making
their own. It `ray.put`s what it receives and holds the ref, so parameters exist once per
node and every Actor there reads them zero-copy; newest overwrites oldest, so a Mirror
that fell behind serves the latest rather than working through a backlog. Publishing
pushes — the Sink writes the Store and then every Mirror, fire-and-forget — and a Mirror
keeps a pull path for cold start and for a restart that came back empty. Whether a Mirror
exists at all is a Topology choice rather than a code change, worth making when Actor
fan-out would otherwise overwhelm the Store, and paid for with a second hop of staleness.
