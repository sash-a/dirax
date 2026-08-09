# brrrl

A distributed reinforcement learning library. Algorithm logic is written as if
single-process; distribution is added by wrapping it, never by changing it.

## Language

### State

**Store**:
The thing that owns a piece of run state that is written and later read back — the
live params. Exactly one exists per run, as an in-process type that a Ray-placed
wrapper holds rather than reimplements.
_Avoid_: Server, cache, manager, registry

**Sink**:
The write half of access to a Store, as a protocol the component holds. Carries a
reference and at most a routing tag, never the state, so a local Store and one
across the cluster are interchangeable to the caller.
_Avoid_: Publisher, writer, emitter

**Source**:
The read half of access to a Store. Same rules as a Sink.
_Avoid_: Getter, provider, client

Only for state with both halves. One-way flows are not Sinks into Stores — see
Metrics, which records into a Collector and never reads back.

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

### Networks

**Torso**:
The part of a network that maps an observation to an embedding. Takes and returns
arrays, so it is shared across algorithms and knows nothing about action spaces.
_Avoid_: Encoder, feature extractor, backbone, body

**Head**:
The part of a network that maps an embedding to an action distribution or a value.
The only part that knows the shape of the action space.
_Avoid_: Output layer, predictor, projection

**Assembly**:
The composition of Torsos and Heads into the network an algorithm applies. Belongs
to the algorithm rather than to the shared primitives, because what it returns and
which params reach an Actor differ per algorithm.
_Avoid_: Model, architecture, network factory

### Learning

**Estimator**:
The rule that turns a Trajectory's rewards and values into the targets and
advantages an Objective consumes. GAE and VTrace are the same kind of thing.
_Avoid_: Advantage module, value estimator, postprocessor

**Objective**:
The rule that turns advantages and an action distribution into the policy's
contribution to the scalar a Learner differentiates. The policy term only — the
value loss and entropy bonus are separate terms weighted alongside it.
_Avoid_: Loss, loss module, criterion

An algorithm is a choice of both, because the two vary independently.

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

### Metrics

**Step**:
The one x-axis every metric in a run is logged against: cumulative Transitions or
steps or env steps produced by Actors. Every other clock — Version, transitions
consumed — is a scalar logged alongside, never an axis.
_Avoid_: log step, iteration

**Metric recorder**:
What a component holds to report metrics. Knows how many Transitions its metrics
account for, never the Step, so a local recorder and one across the cluster are
interchangeable.
_Avoid_: Logger, sink, writer

**Metric collector**:
The one place metrics converge. Owns the Step and routes each Origin to its chain.
_Avoid_: Store, aggregator, hub

**Origin**:
Which component emitted a batch of metrics. Selects its chain and names its
namespace.
_Avoid_: Source, tag, category

**Processor**:
A stage in a metric chain that transforms and forwards — namespacing, windowing.
Implements the same protocol as the backend at the end of the chain.
_Avoid_: Handler, filter, middleware

**Artifact**:
A file or directory belonging to a run — a checkpoint, a video, a config. Rides
the same chain as Metrics and takes the same namespace, but is filed under a
directory rather than logged against the Step: it names a place, not a moment.
_Avoid_: File, asset, output
