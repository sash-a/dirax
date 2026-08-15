# A hand-written tanh-squashed Gaussian

Continuous heads return a `TanhNormal` we implement ourselves, rather than composing a
library's tanh bijector onto its Normal.

**No library gives a usable one.** Entropy is the gap: distrax, numpyro and distreqx all
raise `NotImplementedError` for a tanh-transformed Gaussian, and bare distrax returns NaN
for both `log_prob` **and its gradient** at `|a| = 1` — the values a squashed policy
actually produces. A transformed-distribution wrapper composes the bijector and stops
there, so the arithmetic an entropy bonus needs is in none of them.

**Entropy by Gauss-Hermite quadrature.** Sampling is unbiased but needs a key, which would
leave `Categorical.entropy()` and `TanhNormal.entropy(seed)` with different signatures and
force a branch in every loss. Evaluating the correction at the mode removes the key and is
unusable — measured against reference it **inverts the sign** of `d(entropy)/d(scale)`, so
the bonus would widen the policy without limit. Nine fixed nodes are deterministic, need no
key, and matched reference to within 0.0004 at unit scale where the mode-based estimate was
off by 0.75.

## Consequences

We own these numerics, and a subtly wrong `log_prob` trains fine while learning worse. The
hazards above are regression tests, not comments.

The estimate degrades above `scale ≈ 4`. Node count is the knob if that ever matters.

Clipping to the threshold before the base `log_prob` avoids the expensive `arctanh` path,
so the wrapper measured *faster* than bare distrax — 73 µs against 105 µs on a large batch.
The safety is not paid for in speed.
