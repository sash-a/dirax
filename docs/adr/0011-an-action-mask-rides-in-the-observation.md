# An action mask rides in the observation

An env that masks actions wraps in `ActionMask`, whose observation is a dict: the env's own
observation under `observation`, the legal actions under `action_mask`. A mask describes the
state its observation is of, so it is the same slot's data under ADR-0001 and belongs beside
it rather than in a field of its own. As a leaf of the observation it costs nothing anywhere
else where a Transition field would make every algorithm carry one that only some of them read.

An env that masks nothing carries no mask at all, not an all-true one, and its observation
stays whatever the env already gave. Whether an observation is a dict is then a single
question, asked once where the network is built, and neither kind of run pays for the other.
