# One metrics axis; every other clock is a metric

The step every metric is logged against is cumulative transitions **produced** by
Actors. Nothing else is ever a step. The Trainer's `Version`, and its count of
transitions **consumed**, are logged as scalars inside `Metrics`.

A backend can re-key a series on any scalar it was given, but two series logged
against different steps can never be joined. The axis is the one choice that is
hard to reverse.

## Consequences

One clock means one counter. A component reports how many transitions its metrics
account for and never learns the step; the Metric collector accumulates those
increments and hands an absolute position to a chain. Only Actors report a
nonzero count — everyone else genuinely produces zero, which is true rather than
a sentinel.
