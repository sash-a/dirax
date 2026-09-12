# `every` counts calls of the method it gates

A `Duration` passed to a component means *every n calls of the method it gates*: 20
records, 20 saves, 20 publishes. It doesn't mean transitions, updates, or the metrics axis.
`Schedule` holds the position and spends one tick per ask, so the count is asks since
construction. `EverySeconds` reads the other half of an `Instant`.

The alternative is to advance the count by a number the caller already holds, which
freezes any caller whose number does not move. Only Actors report a nonzero transition
count (ADR-0005), so a window counted in transitions flushes once for the Learner and
never again.

## Consequences

One `EveryTicks(20)` is three different intervals depending on who holds it.
Components wanting different cadences take their own `Duration` rather than sharing one.
A shared `Duration` is safe, but a shared `Schedule` is two callers spending each other's turns.
