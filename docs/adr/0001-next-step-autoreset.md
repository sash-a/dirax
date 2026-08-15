# Next-step autoreset

Gymnasium offers two autoreset modes. We use `NEXT_STEP` (its default) rather
than `SAME_STEP`, because `SAME_STEP` returns an observation belonging to a
*different episode* than the reward and terminal flag beside it, with the true
terminal observation buried in a ragged info dict.

## Consequences

`NEXT_STEP` costs one extra transition slot in the buffer per rollout.
That slot needs masking, use `terminated | truncated`.

That extra transition is the final obs and used to bootstrap if truncated, or ignored if terminal.

A Trajectory is therefore T+1 slots aligned as **observation_t, terminated_t, truncated_t,
reward_{t+1}, action_{t+1}, value_{t+1}, log_prob_{t+1}**. So the flags describe
the observation beside them, not the outcome of the action beside them.
