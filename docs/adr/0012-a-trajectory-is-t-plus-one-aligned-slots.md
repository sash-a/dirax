# A Trajectory is T+1 slots whose flags describe their own observation

A Trajectory is one buffered Timescale of capacity **T+1**, plus `version` as a
single-item one. A slot aligns as **observation, terminated, truncated, reward,
action, value, log_prob**: the flags describe the observation in the slot, and
the reward, action, value and log_prob describe the transition out of it.

This was settled against a run of gymnasium 1.3.0, not from memory — twice it was
settled from a remembered convention and twice that was wrong. With observations
encoding `episode.step` and rewards `10·episode + step`, `NEXT_STEP` gives:

```
step #0  -> obs=1.1  reward= 11.0  term=False
step #1  -> obs=1.2  reward= 12.0  term=True     <- terminal obs arrives WITH the flag
step #2  -> obs=2.0  reward=  0.0  term=False    <- the zero-reward call returns ep-2's FIRST obs
step #3  -> obs=2.1  reward= 21.0  term=False
```

Laid into slots by a Roller carrying `(obs, terminated, truncated)` forward:

| slot | observation | term | reward | action | value |
|---|---|---|---|---|---|
| A | 1.1 | False | 12.0 | a | V(1.1) |
| B | **1.2** Final obs | **True** | **0.0** | *unused* | V(1.2) |
| C | 2.0 ep-2 start | False | 21.0 | a″ | V(2.0) |

What the alignment buys: the unusable slot is flagged by the flags already in it
(shift 0), truncation bootstrapping reads the value in that same slot (shift 0),
and only the recurrent carry reset is shifted, by one. The more common alignment —
flags describing the outcome of the action — puts the Final obs one slot after its
flag, making the mask a shift of one and the carry reset a shift of *two*. A shift
of two is a rule everyone writes as one.

## Why the trailing step folds in rather than getting its own Timescale

Every library that ships Trajectories between processes folds it: torchbeast
stores `T+1` and takes `baseline[-1]`, SEED RL sends `unroll_length + 1`, Acme's
sequences slice `values[:-1]`/`values[1:]`. The ones that keep it separate —
CleanRL, SB3, RLlib, Stoix, Mava — all have the Learner in the same process, where
the trailing timestep is a live variable or sits in a scan carry. Here the
Trajectory is the unit of transfer, so anything outside that pytree never reaches
the Trainer at all.

Folding also deletes the `if t == last:` branch those libraries carry, feeds rlax's
`truncated_generalized_advantage_estimation` its documented `values` of `[T+1]`
directly, and lets a recurrent Learner scan `T+1` steps in one pass instead of
scanning `T` and hand-running the last step with the scan's final carry.

It costs one wasted slot's worth of action and log_prob — about 0.8% of the wire at
`T=128` — and one extra act call per rollout. That act could in principle be saved:
the trailing slot's observation is the next Trajectory's first slot, so the output
would serve twice. It is not worth a Roller carrying a policy output across the
boundary alongside its observation and flags, and the reused value would have been
computed under older params than the slot it landed in.

`version` is a bare `NDArray[np.float32]` of shape `(1,)`. It needs no container
type because ADR-0002 binds only pytree containers, and `Version` is already the
scalar alias. Carrying it inside the Trajectory rather than beside it is what lets a
Trainer log a staleness distribution across a batch instead of two series that
cannot be joined.

Floating point, because it is a **mean**: a Policy adopts new params at whatever
rate it likes, so a rollout can straddle a publish and the slots of one Trajectory
need not share a Version. The alternative — a per-step Version on the buffered
timescale — keeps integers and lets a Trainer take any statistic it wants, at
`T+1` more leaves and a Trajectory whose staleness is no longer one number. Take
that trade only when the spread turns out to matter, which shipping the mean is
enough to find out.

The wire form is the `Trajectory` type, not the accumulator's dict: optree flattens
a sorted-key dict and a field-ordered NamedTuple to different leaf orders, and
echo's `TcpClient` validates shapes and dtype sizes positionally against the
Server's spec. The dict stays an accumulator-local detail, and because its keys are
the Trajectory's own field names the Roller rebuilds the typed container from it.

## Consequences

The remaining shifts move into the Estimator, which is where ADR-0010 already puts
convention translation: the discount reads `~terminated[1:]` and the λ-cut reads
`(terminated | truncated)[1:]`, so a truncation stops the recursion while still
bootstrapping. Every derived boundary signal — unusable slot, bootstrap, carry
reset — belongs in the single learning-module function ADR-0010 mandates, never
open-coded per algorithm. A wrong shift then fails in one place rather than
training slightly worse in several.

The trajectory boundary needs no special case. If the trailing slot carries a set
flag, the next Trajectory's first slot carries the same flags and the env's ignored
step falls on that rollout's first call; the shift-0 mask covers both sides.

No Trajectory carries a `mask` field, and none should be added: the flags are the
mask, and a second encoding of the same fact is a second thing to get wrong.
