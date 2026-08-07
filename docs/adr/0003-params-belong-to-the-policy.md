# Params belong to the Policy, not the Roller

`Policy.act(observation, key)` takes no parameters. The Policy owns its own
param source, and a Roller has no param concept at all. The obvious alternative
— `act(params, observation, key)`, with the Roller holding the source — reads
better in a rollout loop but cannot express an inference server, which owns its
weights and would have to pretend to accept them over the wire.

This is what makes `LocalPolicy` and `InferenceServerPolicy` interchangeable to
any Roller, which is the property the whole acting layer is built around.

## Consequences

A Policy is stateful, so it **cannot live inside a jitted rollout**. Fully
jitted rollers will be a separate family built on pure functions with explicit
params, not an implementation of this protocol. That is a fork in the design,
not a gap to be patched later.

Because params are hidden, the Roller needs the version it acted under. That is
what `Policy.sync()` is for: called once per Trajectory, so a Trajectory has
exactly one Version.
