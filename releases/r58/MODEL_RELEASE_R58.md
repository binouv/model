# FlyGraph R58 policy release

**Status:** CONFIRMED

This is an inference-policy release, not a newly trained weight checkpoint. R58 adds zero trainable parameters and versions the H=1 shallow lookahead policy over frozen R35 + R52.

## Held result

- 64-hop answer: 91.67% vs 83.33%.
- 128-hop answer: 83.33% vs 66.67%.
- 128-hop true pruning losses: 0.000 vs 0.167/episode.

## Model/policy files

- `r58_policy_checkpoint.json` — complete policy hyperparameters + frozen dependency hashes.
- `r35_verifier_weights.npz` — frozen R35 verifier weights.
- `r52_ecc_model.npz` — frozen R52 protected-address codebook.
