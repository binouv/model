# EXT2 2026-10-22: registered before training

One primary hypothesis: repairing instruction/MCQ serialization and including short answer labels in a broader external SFT mixture improves held-out label-ranking accuracy over the existing EXT step256 model without increasing held-out Alpaca NLL. No claim that output-format repair alone improves knowledge. No additional scientific ablation is planned.

Continue the verified 100,028,328 parameter FP32 checkpoint using a fresh AdamW. Same architecture and original tokenizer. Four examples per update: two Alpaca, one OpenBookQA, one PIQA. 4096 updates maximum, 16,384 unique records, one epoch. Train answer/EOS tokens, equal example weights. Checkpoint every64 updates; dev every512; no test during training. Lowest mean dev loss across Alpaca/OpenBookQA/PIQA selects the model; ARC dev is diagnostic only. Prior step0 is eligible, so a regression need not replace it.

Native FP32 parameters/optimizer, BF16 autocast, FP32 delta state; 4CPU and16GiB RAM. Activation checkpointing disabled to avoid recomputation, with unchanged architecture/parameters. Save exact native recovery, RNG, sampler/config/data identity. Keep step64 for exact replay and best/latest recoveries.

Training stops at4096 or the first completed64-update boundary at/after2026-10-23T01:30:00Z. A time stop is explicitly marked budget_limited, not a completed4096 run. Final evaluation and publication deadline2026-10-23T02:35:31Z.

Frozen evaluations: OpenBookQA official test256; PIQA official validation partition reserved as test256, disjoint from dev64; ARC-Challenge official test256 with no ARC training; new Alpaca test128. Old ARC-Easy128 is excluded from training and model selection. Dataset selection uses source records, seed20261022, no model-outcome filtering; one duplicate held question was replaced before any test run, leaving all dev hashes unchanged.

Primary external measure: mean label-ranking accuracy across the three MCQ sets, with per-task raw counts and paired bootstrap confidence intervals versus EXT step256. Report greedy letter accuracy/validity separately and Alpaca held NLL plus raw generations. A convincing improvement requires a positive paired95% interval for mean ranking gain, positive direction on at least two tasks, and no worse Alpaca NLL. This still cannot establish general superiority to missing references.

Both exact reference models remain unmeasured because official Hugging Face CONNECT requests return403. No alternate endpoints, paid services, generated tasks or gold answers. Push only completed, validated evidence and checkpoint bytes to binouv/model at the end, without force.
