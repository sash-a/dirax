# The agent axis exists only where there is a team

Every field an `EnvStepper` reports leads with `(E, A, ...)` for an env with a team acting in it, and
with `(E, ...)` for one with a single agent. A single agent has no agent axis rather than one at
width one: `Agents` is not applied to its env at all, and refuses `num_agents < 2` rather than
accepting one and quietly widening. `prefix(num_envs, num_agents)` is that leading shape, and the
one place the arity is asked.

Filling the axis with a single empty dim is the tidier contract, but it moves complexity into a network
which might be confusing for new users. "Why does my single agent network need to return something with
an extra empty dim?".
