# FlyGraph G1 — 309,993,253 parameter generative bootstrap

**Status: completed finite bootstrap, NOT a pretrained assistant.**

This is a causal next-token language model initialized from scratch, not an earlier restricted-command classifier. All 309,993,253 parameters are trainable. It has 12 Transformer layers with GQA, RoPE, RMSNorm, SwiGLU and tied input/output embeddings, plus a 4,819,237-parameter synthetic recurrent graph readout. The graph performs two rounds independently at each token; it is NOT a biological connectome and does NOT carry recurrent state between tokens. The separate ProtectedMemory API has not been learned as a model tool-use capability.

## Completed run

Seed7101, 192 optimizer steps, batch4, sequence128, **98,304 token presentations**. Full FP32 parameters/gradients, Adafactor. CPU4 cores, 4GiB memory limit, no GPU or external teacher. The local corpus has 3,742,193 CPython/synthetic RU/EN/code tokens; the finite run did NOT process the full corpus.

Validation macro-NLL: initialization9.344726 → step64 4.649470 → step128 3.030272 → step192 **2.652657**. Final held test macro-NLL: **2.704432289** versus **2.704435815** with graph messages zeroed. Relative graph gain: **0.0001304%**, below the preregistered1% criterion. Both arms: **0/21 correct generated answers**, **0/21 strict completions**. No broad reasoning or Qwen-superiority claim.

All14 engineering tests passed; all parameter values finite. Causal-prefix difference0; cache/full-forward difference3.8147e-06. These are correctness tests, not intelligence benchmarks. Forward1024 works but training context was128.

The first attempt was killed by OOM after logging step76. Complete checkpoint64 was recovered with optimizer/RNG/sampler state, then65–192 were executed with activation checkpointing. Uncheckpointed first-attempt steps65–76 are excluded from canonical training metrics.

## Data audit

The old synthetic memory_update subset has a shortcut: its answer always equals the last global write (train3000/3000, validation200/200, test200/200). Language and task family are partially coupled. G1 metrics must only be treated as bootstrap measurements. A separate untrained curriculum prototype removes this specific last-write shortcut and verifies chronological replay; it was NOT used to improve or relabel G1 results.

## Publication scope — do not confuse code with weights

Source, configuration, research and completed-summary metrics are in this directory. **The FP32/BF16 weight bytes, exact tokenizer vocabulary/data, raw local logs and the entire historical C1 archive are NOT all uploaded to GitHub.** Local downloadable bundles preserve them. A manifest is not a weight release. No GitHub Release asset upload has been performed.

FP32 checkpoint:25 safetensors shards,1,239,985,596 weight bytes, plus optimizer/RNG state. BF16 rounded export:10 shards,619,999,114 bytes. All shard hashes verified locally; BF16 reload completed. Exact training resume requires FP32 plus training_state.pt. Do not load untrusted pickle checkpoints.

See `metrics/G1_COMPLETED_SUMMARY.json`, `metrics/G1_METRICS.csv`, `reports/COMPLETED_RUN_RU.md`, and `reports/PUBLICATION_STATUS.json`. Raw local artifacts and exact tokenizer are available in the conversation's downloadable source/data and weight bundles.

## Research and future variants

`research/ARCHITECTURE_REVIEW_RU.md` and `research/SMALL_MODEL_RESEARCH_ADDENDUM.md` record concrete Qwen3.5/DeepSeek/MobileLLM/SmolLM/recurrent-depth ideas. `experiments/g2_gate` is a correctness-tested but UNTRAINED function-preserving headwise gate. `experiments/verified_memory_data` is a verified data-generator prototype, not a model result. Proposed348/546/794M models are designs only.

The namespace deliberately does not overwrite the older, differently numbered ECC R52–R61 experiments in this repository.
