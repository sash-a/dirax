# An action mask rides in the observation

An env that masks actions wraps in `ActionMask`, which puts the legal ones in the
observation beside it. An env that masks nothing carries no mask at all — not an all-true
one — and PPO reads a plain observation.

## Not a Transition field

A mask describes the state its observation is of, so it is the same slot's data under
ADR-0012, and putting it there costs nothing anywhere else: `example` sizes it from the
observation space, the accumulator writes it as another leaf, and a Roller, `echo` and
`PPOUpdateRule` are untouched. A field on `PPOTransition` would have needed all four to
change, and every other algorithm to carry a field only this one reads.

`PPOUpdateRule` never sees a mask. It hands `apply_fn` the whole observation, and which
part of that reaches which Head is the Assembly's business.

## One Assembly, one question

Whether an observation carries a mask and whether the Head reads one are the same fact, so
`PPONetwork` asks it once:

```python
masked = isinstance(self.policy, MaskedDiscreteHead)
emb = self.torso(obs[OBSERVATION] if masked else obs)
return (self.policy(emb, obs[ACTION_MASK]) if masked else self.policy(emb)), self.value(emb)
```

That one boolean is what keeps every other layout decision from multiplying. Masked and
unmasked runs are otherwise the same network, the same Trajectory and the same loss, and
neither pays for the other: an unmasked run carries no mask and does no `where` over its
logits.

ADR-0008 refuses Mava's wart, where `ContinuousActionHead` takes an `action_mask` purely to
match the discrete signature and immediately discards it — forced on it by a `base.py` that
calls every Head uniformly. A generic Assembly is exactly where that pressure comes from,
so this one answers it by branching in the Assembly rather than in the Heads. The branch is
Python, resolved when flax traces the module, and a Head still takes exactly the arguments
it uses.

## An entropy is read against the legal actions

`uniform_entropy` counts a categorical's finite logits rather than its categories, so a
policy uniform over what a mask leaves reads as one rather than as the fraction of the
action space that happened to be legal. Without it the metric tracks the mask instead of
the policy: on Taxi, whose states leave about three of six actions legal, a fresh policy
reads 0.61 and looks half collapsed. An unmasked categorical has every logit finite, so it
is the same count either way and the two runs report on the same scale.

It is per state, so the normalisation happens before the mean over the batch rather than
after. A state leaving exactly one action legal has zero entropy out of a possible zero and
reads as one, because a forced move says nothing about whether the policy has collapsed.

## Every state must leave one action legal

A state that leaves none has no policy at all. Its entropy is nan, and because the network
runs on every slot, that nan reaches the gradient through slots the loss has already masked
off — a Final obs included. Nothing defends against it: it is an environment to fix, and
the loudest possible failure is the right one.
