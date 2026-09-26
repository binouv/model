# G4 Prefix-Hybrid — preregistered mechanism test

G3 completed nine small generative trainings. G4 changes one factor: the two attention sublayers may attend bidirectionally within the fully observed input prompt; answer generation remains causal. Control is the same hybrid with causal prompt attention. Both arms have identical parameter count, data, objective, batches and optimizer schedule.

Three seeds 7501/7502/7503, 1200 updates, batch32, fixed final checkpoints. This is a 469,648-parameter from-scratch mechanism probe using a lossless UTF-8 byte vocabulary, not a G3 continuation, 300-800M result, or Qwen parity claim.
