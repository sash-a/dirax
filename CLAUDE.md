## brrrl
This is a library for distributed reinforcement learning. It aims to be highly optimised and as fast as possible, while remaining understandable and hackable.
It flexible on distribution strategies and hardware. It has different RL algorithms, both on- and off-policy.

## Tech stack
Placement is done with ray.
Communication is done with ray off the hot-path and custom/sockets/zmq where required.
Learning is done with JAX.
Neural networks in Flax.

## Coding style
We only have final classes (no inheritance). We use traits/interfaces instead.
We strongly favour composition for decoupling and code re-use.
If you have to write a long comment to describe code you have written it poorly.

## Agent skills

### Issue tracker

Issues and specs live as GitHub issues, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, each label string equal to its name. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
