# G7R — fresh multi-family trace curriculum

Continuation of the already preregistered G7R protocol. Six fixed 2,000-update models, three seeds per arm, 469,648 parameters. Trace→final uses verified compact traces for steps1–1000 then final-only; control is final-only throughout. Same semantic minibatches and initialization within seed. Fresh semantic groups are disjoint from the denied parent groups. Held never selects checkpoints.

This is a small English synthetic mechanism probe, not 300–800M training, not broad code generation, and not Qwen parity. Exact data hashes and success gates remain those in configs/preregistered.json. The code added here is committed before execution. Native checkpoint bytes are retained locally if a release-asset write path is unavailable; text results and hashes are pushed non-force.
