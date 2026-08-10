# An algorithm is an Estimator and an Objective

The on-policy family factors along two independent axes, and the pair is made visible
where the Learner is constructed rather than sealed inside one class per algorithm.

| | Estimator | Objective |
|---|---|---|
| A2C | n-step / GAE | policy gradient |
| PPO | GAE | clipped surrogate |
| IMPALA | VTrace | policy gradient |
| APPO | VTrace | clipped surrogate |

APPO stops being a new algorithm and becomes a cell in the grid, which matters because
ADR-0008 already commits us to writing it.

This is ADR-0008's ruling carried from networks to maths: primitives are shared,
composition belongs to the algorithm. There is no `PPOLoss` type and no base class
beneath one. torchrl found the same decomposition — their IMPALA recipe is literally
`A2CLoss + VTrace` — but kept the pieces as classes configurable enough to serve every
caller, which is how `objectives/ppo.py` reached 1,813 lines.

## rlax supplies the primitives, for now

It already has all of them, and ADR-0007 commits us to the distrax/tfp tree it sits on.

The cost is that rlax expresses an episode boundary as a single `discount_t`, while we
keep `terminated` and `truncated` apart and ADR-0001 adds a Final obs slot. The conventions
reconcile — GAE's per-timestep `lambda_` carries truncation, vtrace's `sample_mask`
carries the unusable slot — but that is not the call anyone writes by default, and getting it
wrong still trains, just worse. So the translation is **one function in `learning/`**,
never open-coded per algorithm.

If that function stops being small — a second convention for off-policy targets, a
signature that cannot express the Final obs — rlax is buying less than it costs. Then we
own the arithmetic and keep rlax as a **test oracle** asserting our GAE and VTrace match
theirs. That is ADR-0007's shape: take the library for what it does well, write the part
that matters, measure against reference.

## Consequences

An algorithm's identity is readable at its construction site — a reviewer can tell PPO
from APPO without opening a loss. The value-loss and entropy coefficients sit there too;
Objective names the policy term only.

Nothing stops an Estimator being paired with an Objective that does not belong with it.
The table above is the test matrix.
