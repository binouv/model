# FlyGraph G7R completed local grid

Verdict: NOT_CONFIRMED_IN_REGISTERED_PROTOCOL.

Six fixed-step 2000-update, 469648-parameter models completed. Trace-to-final IID counts were 9/12/8 of160 versus 3/3/2 for final-only: 6.042% versus1.667%, gain +4.375 pp; the registered gate required +5 pp. Surface gain +0.417 pp and extrapolation -0.417 pp, so the registered pooled transfer gate also failed.

The effect is localized to list/memory, not general reasoning. Trace-to-final arithmetic correct by seed:0/0/1 of32; code-trace:0/0/1. Exact canonical trace after final consolidation:0/160 for all3seeds.

All six final safetensors reload exactly; 96 generated IID cases replay with zero mismatch. Native checkpoints include model, AdamW, Python/torch/sampler RNG at steps500/1000/1500/2000, and all six next-update replay checks passed. Re-executed trace step500 safetensors are byte-identical to all three earlier partial hashes.

Actual checkpoint bytes are preserved in local downloadable bundles. No remote release-asset write action is available in the connected GitHub tool during this run, so no remote weight release is claimed.
