# distrax, with a hand-written squashed Gaussian

Heads return `distrax` distributions. distrax depends on `tfp-nightly`, which we accept,
and we write the tanh-squashed Gaussian ourselves, which we cannot avoid.

**Why the nightly is acceptable.** Stable `tensorflow-probability` is broken against the
JAX we require — `jax.interpreters.xla.pytype_aval_mappings` no longer exists — which is
why other JAX RL libraries are pinned to much older JAX. The nightly is the only working
backend, not a preference. It pins fine: PyPI holds ~2,950 `tfp-nightly` builds going
back to March 2018 and prunes none of them, so a lockfile stays resolvable.

**Why we write the distribution anyway.** No library implements entropy for a
tanh-transformed Gaussian. distrax, numpyro and distreqx all raise `NotImplementedError`,
and bare distrax returns NaN for both `log_prob` **and its gradient** at `|a| = 1`. So the
dependency contributes `Categorical` and the arithmetic we would have written regardless.

Ported from Mava's version, two things change:

- Mava overrides tfp's private `_parameter_properties`. distrax has no such hook, so that
  workaround disappears.
- It must sum over the action axis itself rather than nesting inside `distrax.Independent`,
  whose `entropy()` drops kwargs where tfp's forwards them. Nesting raises a `TypeError`
  on the seed.

**Entropy by Gauss-Hermite quadrature.** Sampling is unbiased but needs a key, which would
leave `Categorical.entropy()` and `TanhNormal.entropy(seed)` with different signatures and
force a branch in every loss. Evaluating the correction at the mode removes the key and is
unusable — measured against reference it **inverts the sign** of `d(entropy)/d(scale)`, so
the bonus would widen the policy without limit. Nine fixed nodes are deterministic, need no
key, and matched reference to within 0.0004 at unit scale where the mode-based estimate was
off by 0.75.

## Consequences

We own these numerics, and a subtly wrong `log_prob` trains fine while learning worse. The
three hazards above are regression tests, not comments.

The estimate degrades above `scale ≈ 4`. Node count is the knob if that ever matters.

Clipping to the threshold before the base `log_prob` avoids the expensive `arctanh` path,
so the wrapper measured *faster* than bare distrax — 73 µs against 105 µs on a large batch.
The safety is not paid for in speed.

We carry 7 MB and a 1.8 s import in every process, including every Ray actor. In exchange,
a user can return any distrax distribution from their own Head and our algorithms consume
it unchanged.
