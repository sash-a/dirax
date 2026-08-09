"""distrax works against the JAX this repo requires.

ADR-0007 accepts a nightly ``tensorflow-probability`` because the stable release is broken
against that JAX. This is the gate on that claim: until a distribution is constructed here
and returns finite numbers, every Head is building on an unverified bet.
"""

import distrax
import jax
import jax.numpy as jnp

NUM_ACTIONS = 5


def test_categorical_answers_the_four_calls_a_loss_makes(leading: tuple[int, ...]) -> None:
    distribution = distrax.Categorical(logits=jnp.zeros((*leading, NUM_ACTIONS)))

    action = distribution.sample(seed=jax.random.key(0))
    log_prob = distribution.log_prob(action)
    entropy = distribution.entropy()
    mode = distribution.mode()

    for result in (action, log_prob, entropy, mode):
        assert result.shape == leading
        assert jnp.isfinite(result).all()
    assert ((0 <= action) & (action < NUM_ACTIONS)).all()
