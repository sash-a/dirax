# Algorithm files run without Ray

Each file in `algorithms/` carries a `main()` that trains in one process on one
thread, and imports nothing from `distributed/`. `python -m brrrl.algorithms.ppo`
must train CartPole with no Ray anywhere in its transitive imports.

This turns the layering from a convention into a CI check: if an algorithm can
no longer run standalone, distribution has leaked into it. It also gives the
debugging story that matters — reproduce a learning bug in a debugger, without a
cluster.

## Consequences

`main()` writes its own sequential loop rather than reusing the `Actor` and
`Trainer` components, whose loops genuinely differ — they block on queues, batch
across actors, and handle staleness. Roughly eight lines of orchestration exist
in both places.

That duplication is deliberate. `main()` is an executable spec of the algorithm
API, and it gives a standing constraint worth keeping: **if the training loop
can't be written in about eight readable lines, the protocol surface is too
big.** Parity between the two paths is a test — both must learn CartPole — not
something bought with shared code.
