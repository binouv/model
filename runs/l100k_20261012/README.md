# L100K: neural byte-record memory circuit for FlyGraph

Strict experimental provenance: `configs/registered.json`, GitHub `runs/l100k_20261012/REGISTRATION.md` committed before training. This is a **narrow trained adjunct memory circuit**, not an independently capable replacement for Qwen3.5‑0.8B, and it does not update the existing 100M model weights.

## Reproduce

```bash
python -m pip install torch safetensors pytest
pytest -q tests
python src/train.py --train
python src/train.py --evaluate
```

The text serialization contract is `a=120;b=45;a=7?b`. The inference receives only raw byte input and the trained circuit. The source-position pointer is fully learned, while extracting all following digits is an explicitly **engineered copy operator** (not unconstrained autoregressive language generation). The reference is a structurally query-blind arm, **not** a Qwen model.

Checkpoints every 64 complete updates include FP32 module safetensors, native AdamW moments, Torch/Python RNG, sampler state, complete update number, model/config/data SHA256. `in_progress` checkpoints may be used for resumption but do not establish a completed scientific result. No CUDA or external reference inference is claimed.