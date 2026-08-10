# Next-step autoreset

Gymnasium offers two autoreset modes. We use `NEXT_STEP` (its default) rather
than `SAME_STEP`, because `SAME_STEP` returns an observation belonging to a
*different episode* than the reward and terminal flag beside it, with the true
terminal observation buried in a ragged info dict — a transition that looks
valid and isn't.

## Consequences

`NEXT_STEP` costs one Trajectory slot per episode per env: the step whose action
the env ignores and whose reward it zeroes. That slot needs masking, but no
`mask` field. Because a Transition carries the flags describing its own
observation, `terminated | truncated` on the **same slot** already says so.

That slot holds the **Final obs**, and it is real. Truncation bootstrapping reads
its value from there; only a termination reads nothing from it. Don't "optimise"
the slot away.

The new episode's first observation lands one slot further on, so a recurrent
carry resets on `terminated | truncated` at `t-1` — one shift later than the
mask, and the only shift the layout has.
