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

**Mirror**:
A Source holding a copy of a Store's state, refreshed on its own cadence so many
readers share one round trip. Interchangeable with the Store to a reader, and adds
a second hop to staleness.
_Avoid_: Cache, replica, proxy, local store

Only for state with both halves. One-way flows are not Sinks into Stores — see
Metrics, which records into a Collector and never reads back.

### Data

**Transition**:
One environment step: an observation and whether its episode ended there, the
action taken from it, and what that action earned.
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

**Final obs**:
The observation of the state an episode ended in, carried by the same Transition
as the flag that says so. Truncation bootstrapping reads its value; a
termination reads nothing from it. Either way it is the action and reward beside
it that are meaningless, never the observation.
_Avoid_: Dummy step, padding, invalid step, reset step

### Acting

**Roller**:
Produces one Trajectory by stepping an environment with a Policy. Owns the
algorithm's rollout logic and knows nothing about where anything else lives.
_Avoid_: Collector, sampler, worker

**Policy**:
Turns an observation into an action. Owns its own parameters, so a local policy
and an inference-server client are interchangeable to a Roller.
_Avoid_: Agent, network, model

**Env stepper**:
What a Roller steps to advance its environments, presenting one step API with a
leading env axis on every field. Reports its finished episodes beside the step,
so a Roller reads them the same way whatever built the environments.
_Avoid_: Env, environment wrapper, vectoriser

**Sync point**:
The moment a Policy adopts the latest parameters. When to take one is the
Policy's own business — as often as every act — so a Trajectory reports the mean
Version its acts ran under rather than one taken on the way in.
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

### Components

**Component**:
A class Ray places as its own process. Declares the Ray behaviour options its own
correctness depends on, and nothing about where it lands — that is the caller's
Placement.
_Avoid_: Remote, worker, service

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
envs. Owns its parameters, so its clients never handle them.
_Avoid_: Policy server, actor server

**Topology**:
A named arrangement of Actors, Trainers, and inference servers — how many of
each and how they are wired. Says nothing about which host anything lands on, so
one Topology runs on any cluster.
_Avoid_: Architecture, layout, config

**Placement**:
What one role's processes ask the scheduler for, and where they may land.
Supplied by the caller, because only their cluster knows what a name like
"learner" means. Empty by default, so a run that names no Placement lands
wherever Ray will take it.
_Avoid_: Resources, slot, allocation

**Gang**:
A set of Components that must be placed together or not at all — Trainer shards
forming one JAX process group, or an inference server and the Actors it serves.
Reserved as one unit, and each member's position in it is that member's stable
identity. Actors at large are not a Gang: a run with fewer of them works, just
slower.
_Avoid_: Group, cohort, bundle

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
