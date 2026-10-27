# Placement is the caller's; behaviour is the class's

dirax never names a cluster resource. A `Placement` carries what one role's processes
claim and where they may land, it is supplied by the caller, and every field defaults to
unset — so a run that names no Placement lands wherever Ray will take it. The Ray options
that encode a Component's own correctness are declared on the Component instead, and the
two meet in one call.

Ray exposes 26 actor options. They divide three ways, and the division is the whole
decision:

| | owner | where it lives |
|---|---|---|
| **Claim** — `num_cpus` `num_gpus` `memory` `resources` `label_selector` `accelerator_type` `env` | caller | `Placement`, a frozen dataclass |
| **Behaviour** — `max_concurrency` `max_restarts` `max_task_retries` `allow_out_of_order_execution` `max_pending_calls` | the class | `RAY_OPTIONS: ClassVar[Behaviour]` |
| **Identity and derived env** — `runtime_env`, `name`, `namespace` | the driver | computed at the placement site |

`Placement.apply(component, group=None, index=-1)` merges the three and takes nothing
else. `Component` is a one-member Protocol — `ty` rejects a class with no `RAY_OPTIONS`
through `type[Component]`, so the contract is checked rather than remembered.

## Why the claim set is closed

`Behaviour` is a `total=False` TypedDict rather than an open mapping, and `Placement` has
no `**extra`, because the excluded options are the ones that break Components silently.
`CheckpointStore` is fire-and-forget on the strength of Ray executing its calls in order;
`max_task_retries=1` breaks that, since retried actor tasks are always executed out of
order. A restarted `ParamStore` comes back empty and every Policy acts on stale params
without an error anywhere. Those defaults are load-bearing, so they are declared, and a
config file cannot reach them.

The closed set costs something real. A cluster handing out MIG slices as a custom
resource must claim `num_gpus` fractionally instead. We took that over an escape hatch
whose only unique power is reaching the options above.

## Why the caller names the resources

`resources={"learner": 1}` means something only because a manifest minted the string. A
library that writes it hardcodes one cluster's vocabulary, and then needs an environment
variable to fake that vocabulary locally. Deleting the hardcode deletes the mock: on a
laptop `Placements()` claims nothing, and `ray.init()` already reads `RAY_ADDRESS` itself,
so dirax neither wraps nor calls it and ships no cluster detection at all.

Docs lead with `label_selector` over custom `resources` because custom resources are
*consumed*. A learner node advertising `learner: 1` can host the Trainer or the
CheckpointStore, not both — the second sits pending forever. Labels are non-consumable
node attributes, which is what "this is the learner box" actually means; `num_cpus`
against the node's real capacity is what limits how many fit.

## Why counts are explicit, and why waiting is narrated

Actor count stays in the Topology even though the cluster manifest states it too. Deriving
it would make capacity into intent: `echo`'s Server batches across whoever connected, so a
run short four Actors trains to completion at 93% throughput with a shifted staleness
distribution and no line of output anywhere. Actor count sets batch composition and
staleness, which are the things this library exists to get right — it is a hyperparameter
wearing infrastructure's clothes, and two runs with identical configs must not differ
because a cluster came up small.

Stating it twice is only safe if the two can be checked against each other, so every
component is waited on via `__ray_ready__` (which Ray injects on every handle, and which
stays pending while unschedulable) and every gang via `PlacementGroup.wait`. The wait is
narrated — progress while pending, then failure at 600s naming the unsatisfied claim
beside the per-node `Resources` and `Labels` from `ray.nodes()`. A deadline alone would be
wrong at both ends: queueing twenty minutes for capacity on a shared cluster is normal,
and an unbounded silent wait is the failure this exists to remove.

## Gangs reserve; they do not wrap

A **Gang** is a set of Components that must be placed together or not at all — Trainer
shards forming one JAX process group, or an inference server and the Actors it serves.
Actors at large are not one: a run with fewer works, just slower.

`reserve(placement, size, strategy)` derives `bundles` and `bundle_label_selector` from
the claim and returns a **raw** `PlacementGroup`. Deriving them is the point — state a
bundle's resources separately from the member's claim and a mismatch means the member does
not fit its bundle and hangs. Returning a raw group is also the point: a hand-built group
with heterogeneous bundles passes to `apply` unchanged, so none of Ray's placement surface
is fenced off. There is no `Gang` type; the concept is vocabulary, the mechanics are two
functions.

Because a live `PlacementGroup` is a runtime handle it cannot be serialised into a config,
which settles where it goes without appeal: **the claim is config, the group is runtime**.
Groups arrive through `scheduling_strategy=PlacementGroupSchedulingStrategy(...)`, the
direct `placement_group=` options being deprecated — and since `scheduling_strategy` holds
one value, per-role `SPREAD` is a gang strategy rather than a Placement field.

## Considered and rejected

- **Deriving Actor count from the cluster.** Incompatible with the unset default (there is
  no answer to "how many fit in an unbounded claim"), defeats autoscaling (an autoscaler
  grows in response to stated demand, and this states none), and requires reimplementing
  Ray's scheduler to know what fits. A `Fill()` sentinel remains available as a purely
  additive opt-in if stating the count twice proves to grate; it would still only work on
  a static, homogeneous, one-role-per-node cluster.
- **One placement group for the whole run.** Bundles may not be empty, so with unset local
  defaults every bundle would be `{}`; synthesising `{"CPU": 1}` instead needs 60 CPUs on
  a laptop and hangs. Atomic startup is what the preflight already reports.
- **`Placements` as an open `Mapping[str, Placement]`.** A misspelled key is unreadable by
  anyone and leaves that role unplaced — the exact silence this ADR removes elsewhere.
  Third-party Components, heterogeneous Actor pools and an evaluator role are all additive
  from a closed dataclass.
- **A `ray_init()` wrapper with environment detection.** Its three jobs are gone: the
  resource mock is unnecessary, `RAY_ADDRESS` is read by Ray, and `runtime_env` excludes
  apply only to a `working_dir` upload.
- **Deriving `XLA_PYTHON_CLIENT_MEM_FRACTION` from a fractional GPU claim.** A fractional
  `num_gpus` is scheduling weight, not a memory budget. Ray does not partition the device, so
  every process sharing a card still sees all of it, and `num_gpus=0.5` says nothing about
  whether that process wants half the memory or 80% of it. Deriving one from the other caps a
  process at a number nobody chose, and the failure — an OOM inside XLA — reads nothing like
  its cause. A Trainer and an inference server sharing a card do still each need a fraction
  set or they will both try to take 75%; `Placement.env` is where the caller sets it, per
  role and exactly.

## Consequences

`CheckpointStore`, `RayMetricCollector`, `ParamStore` and `InferenceServer` lose their
`@ray.remote` decorators — a decorated class cannot be instantiated at all, so this is
what makes them unit-testable — and each declares `RAY_OPTIONS` instead.

dirax owns the JAX environment of every process it places, because Ray leaves
`CUDA_VISIBLE_DEVICES` *unset* for a `num_gpus=0` actor rather than blanking it, and JAX
then preallocates most of a card it was never given. A claim of no GPU derives
`JAX_PLATFORMS=cpu`, and that is the whole of the derivation. `Placement.env` overrides it,
and job-level `env_vars` from the caller's `ray.init` survive underneath, Ray merging per
key. Values a config file cannot hold — a run id, a coordinator address — reach a claim
through `Placement.with_env`, which derives a new claim rather than adding a second merge
path.

A Component's behaviour must be a constant. `InferenceServer`'s `max_concurrency`
plausibly scales with the size of the cohort it serves, and would be the first case that
is not — left to surface rather than designed for.

`jax.distributed` coordination is deliberately out of scope here and belongs in its own
ADR: who is process 0, whether the Learner shards or replicates, and what a dead shard
means are separate decisions. This one only guarantees the input they need, a stable
per-member identity, which a bundle index provides and Ray's scheduler does not.
