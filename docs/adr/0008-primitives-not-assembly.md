# `networks/` holds primitives; algorithms own Assembly

`networks/` holds Torsos, Heads, distributions and the recurrent cell — array in, array
out. It does not hold `FeedForwardActor`, `RecurrentCritic` or anything that composes a
Torso with a Head. Each algorithm owns its own Assembly.

Mava and Stoix both centralise Assembly in a `base.py`, so this is a deliberate
departure. It is driven by the fact that Assembly is not shared in the first place:

- **PPO's** Roller needs a value at act time to compute advantages, so its Assembly runs
  the critic on every act call and critic params must reach Actors.
- **IMPALA's** Learner recomputes values from observations, so shipping critic params to
  its Actors is pure waste.
- **APPO's** optional KL term needs the behaviour policy's full distribution parameters
  in the Trajectory, where PPO stores only the sampled log-probability.

Same Torsos, same Heads, three different things crossing the wire. A shared Assembly
would have to be configurable along exactly the axes that differ, which is how a
`Catalog` happens.

Heads still return **distributions**, and Assembly returns a distribution and a value.
The distribution survives into the Learner, so a loss reads the way a single-process one
would. It collapses to arrays at exactly one place — the act function — because that is
the only point that crosses a process boundary. An inference server cannot return a
distribution usefully: it would ship parameters and have the client sample, moving
compute off the accelerator the server exists to use.

## Why `networks/` exists at all, rather than `examples/`

Distributed RL libraries split cleanly, and not along the axis you would guess. It is not
distributed-versus-not — it is whether the library ships algorithms or only machinery.

```
moolib          RPC/transport only        no networks; nets live in examples/vtrace/
torchbeast      one reference IMPALA      AtariNet defined inline in monobeast.py
SEED RL         4 agents, 4 env families  networks.py per env dir, no shared module
ACME            many agents               acme/jax/networks/ + per-agent networks.py factory
RLlib           many algorithms           rllib/core/models/ — Encoder / Head / Catalog
Sample Factory  distributed APPO          model/ + a model_factory registry
```

brrrl ships a family of algorithms, which puts it with ACME rather than with moolib. All
four primitive modules are identical across PPO and IMPALA, so none of them is
algorithm-specific enough to push down.

ADR-0004 settles the rest. `python -m brrrl.algorithms.ppo` must already train CartPole
standalone, so **`algorithms/` is the examples folder**. A separate `examples/` would
either be imported by the package, which is backwards, or duplicate what `algorithms/`
already runs.

## Consequences

Primitives take arrays and nothing else, so Stoix's `inputs.py` adapter layer has no
reason to exist. Each algorithm unpacks its own observation type.

Mava's wart disappears by construction: its `ContinuousActionHead` accepts an
`action_mask` purely to match the discrete signature and then does `del action_mask`,
which is forced by `base.py` calling every Head uniformly. With no generic Assembly,
Head signatures are free to differ, and masking is its own final Head type.

Every algorithm writes its own small composition module and there will be near-duplicate
ones. That is the same trade ADR-0004 already took for `main()`, and for the same reason:
the duplicated code is the part that differs.

A Head must keep its distribution's parameters reachable rather than hiding them behind
sampling — otherwise APPO cannot store behaviour logits and cannot be written at all.
