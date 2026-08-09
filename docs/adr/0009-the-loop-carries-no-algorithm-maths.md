# The loop carries no algorithm maths

ADR-0004 asked the training loop to fit in about eight readable lines. That rule is
dropped. `main()` may wire as many collaborators as the algorithm needs; the constraint
is on *what* the loop contains rather than how long it is. The loop orchestrates, and
every piece of algorithm arithmetic lives inside `Learner.update`.

The loop is the fork line. `main()` steps a Roller and a Learner in lockstep, and the
distributed path cannot — the Roller sits on an Actor, the Learner on a Trainer, and a
queue with staleness in between. ADR-0004 already conceded that those loops genuinely
differ. So whatever sits in the loop is written twice, and whatever is written twice can
drift silently, in a way that still trains.

Construction does not fork. `PPOLearner(advantage=..., loss=..., optimiser=...)` is
identical in both paths, so that is where the glue is made visible. Exact API is tbd.
## Consequences

`main()` grows past eight lines, and that is fine. The standing check moves from length
to content: if a diff adds arithmetic over rewards, values or log-probabilities to a loop
rather than to `Learner.update`, it is in the wrong place.

The Trainer keeps the property `CONTEXT.md` claims for it — consumes stacked
Trajectories, runs the Learner, contains no algorithm logic — by construction rather than
by discipline.
