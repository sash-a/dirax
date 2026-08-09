# Next-step autoreset

Gymnasium offers two autoreset modes. We use `NEXT_STEP` (its default) rather
than `SAME_STEP`, because `SAME_STEP` returns an observation belonging to a
*different episode* than the reward and terminal flag beside it, with the true
terminal observation buried in a ragged info dict — a transition that looks
valid and isn't.

## Consequences

`NEXT_STEP` costs one Trajectory slot per episode per env: the step after a
terminal has its action ignored and its reward zeroed. That slot needs masking,
but no `mask` field — `terminated | truncated` at `t-1` already says so.

The dummy step's **observation is real**. It is the terminal observation, and it
is where truncation bootstrapping reads its value from. Don't "optimise" the
slot away.
