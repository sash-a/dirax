# `networks/` holds primitives; algorithms own Assembly

`networks/` holds Torsos, Heads, distributions and the recurrent cell — array in, array
out. It does not hold `FeedForwardActor`, `RecurrentCritic` or anything that composes a
Torso with a Head. Each algorithm owns its own Assembly.

Heads still return **distributions**, and Assembly returns a distribution and a value.
The distribution survives into the `UpdateRule`, so a loss reads the way a single-process one
would. It collapses to arrays at exactly one place — the act function — because that is
the only point that crosses a process boundary. An inference server cannot return a
distribution usefully: it would ship parameters and have the client sample, moving
compute off the accelerator the server exists to use.

## Consequences

Primitives take arrays and nothing else, so each algorithm unpacks its own observation type.

Every algorithm writes its own small composition module and there will be near-duplicate
ones. That is the same trade ADR-0004 already took for `main()`, and for the same reason:
the duplicated code is the part that differs.

A Head must keep its distribution's parameters reachable rather than hiding them behind
sampling.
