# brrrl

A distributed reinforcement learning library. Algorithm logic is written as if
single-process; distribution is added by wrapping it, never by changing it.

## Language

### Data

**Transition**:
One environment step: what the agent saw, what it did, and what came back.
_Avoid_: Timestep, experience, SARS tuple

**Trajectory**:
What one Roller sends to the Trainer — a fixed number of Transitions plus any
trailing context the algorithm needs. The unit of transfer.
_Avoid_: Rollout, episode, batch, sample

**Timescale**:
A named group of leaves inside a Trajectory that fill at the same rate. Every
timescale carries a leading time axis; ones holding a single trailing value have
`T = 1`.
_Avoid_: Section, group, field

**Lane**:
One independent environment stream inside a Roller. Lanes are the second axis of
every Transition leaf, so a single environment is one lane rather than a special
case.
_Avoid_: Worker, env index, batch element

**Dummy step**:
The step immediately after an episode ends, where the action is ignored and the
reward is zero. Its observation is real — it is the terminal observation, and
where truncation bootstrapping reads from.
_Avoid_: Padding, invalid step, reset step

### Acting

**Roller**:
Produces one Trajectory by stepping an environment with a Policy. Owns the
algorithm's rollout logic and knows nothing about where anything else lives.
_Avoid_: Collector, sampler, worker

**Policy**:
Turns an observation into an action. Owns its own parameters, so a local policy
and an inference-server client are interchangeable to a Roller.
_Avoid_: Agent, network, model

**Sync point**:
The moment a Policy aligns with the latest parameters and reports the Version it
is now acting under. Once per Trajectory, so a Trajectory has exactly one
Version.
_Avoid_: Refresh, pull, update

### Parameters

**Param source**:
The read side — where a Policy gets current parameters.
_Avoid_: Param server, store, cache

**Param sink**:
The write side — where a Trainer publishes parameters after an update.
_Avoid_: Param server, publisher

**Version**:
A monotonic count of Trainer updates, stamped on parameters. The difference
between the Version a Trajectory was acted under and the Trainer's current
Version is its staleness.
_Avoid_: Step, iteration, epoch

### Placement

**Actor**:
Runs Rollers and ships their Trajectories. Contains no algorithm logic — only
the placement and communication of Rollers.
_Avoid_: Worker, rollout worker, sampler

**Trainer**:
Consumes stacked Trajectories, runs the Learner, and publishes parameters.
Contains no algorithm logic.
_Avoid_: Learner, optimiser, updater

**Learner**:
The algorithm's update rule — a pure function from state and a stacked
Trajectory to new state and Metrics. Distinct from the Trainer, which places it.
_Avoid_: Trainer, loss, update step

**Inference server**:
A Policy hosted in its own process, batching act requests across many Actors'
lanes. Owns its parameters, so its clients never handle them.
_Avoid_: Policy server, actor server

**Topology**:
A named arrangement of Actors, Trainers, and inference servers across devices
and hosts.
_Avoid_: Architecture, layout, config
