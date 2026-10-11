# L100GE (2026-10-12) — preregistered general 100M generative readout-adapter pilot

**Not a completed result at registration.** Base: physically verified L100EC curriculum BF16 model, 100,028,328 parameters, SHA256 `07e33eb2c6aab5727fbb72c0b403bc83ca8e5695d379106af089b27baf995b25`; frozen base is restored strictly from actual checkpoint. A learned post-final RMSNorm low-rank residual (768→8→768; 12,288 FP32 parameters) changes the LM's logits. Unlike L100K, this experiment actually trains an adapter in the **generative model inference path**.

Hypothesis: direct teacher-forced bilingual synthetic multi-task adaptation improves fixed held free-generated exact-answer accuracy over frozen parent, with no symbolic tool/solver or external answer candidates.

Fixed protocol: AdamW lr=0.001, betas(.9,.95), no decay, grad clip1; seed121220, sampler121221; 256 fully completed optimizer updates, effective batch4 via microbatches2; save exact weights/optimizer/RNG/sampler every64 and final. BF16 autocast CPU with original FP32 delta operator, no GPU available. Train exclusively from already frozen L100A `train_memory.jsonl` and `train_replay.jsonl`; 50% memory / 50% replay each minibatch. Canonical L100A held group IDs excluded from train. Fixed held endpoint: the first60 `regression.jsonl` and first100 `iid.jsonl` rows. No held-based tuning. Greedy24 tokens, full8192 vocabulary, strict `F=<integer>` plus EOS; compare frozen L100EC and adapted model on the identical prompts and record all raw generations. Gate: +3 percentage points on pooled160 held questions. Eval and all failures to be published, not Qwen parity. No independent external general-language benchmark and no new Qwen run; broad Qwen3.5‑0.8B superiority remains **unachieved**. Existing L100DA and L100CS training results neither modified nor conflated.

Data hashes:
- `train_replay.jsonl` SHA256 `2e63cac3c4376f7635e36de9d9158f27baba6cd22f5d61eaa8a5d02576e74907`;
- `train_memory.jsonl` SHA256 `5bc497d125fddca9e8b96300f98aef2b5fd15d454a15fe6790de8041633ed28e`;
- `regression.jsonl` SHA256 `5337a961ddc4c383117005bbd42e19dff4a5b83d677c4ef992f14c23be189577`;
- `tokenizer.json` SHA256 `aa6f3b462d4d1bb9ace09d0421a1879500fea91127a347a29df3e46a6f58c500`.
