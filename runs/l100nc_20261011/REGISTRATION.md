# L100NC — natural-language code continuation

Registered before scientific training on 2026-10-11. This is not a completed result or a claim of Qwen superiority.

Parent: actual mounted L100EC curriculum BF16 weights, 200069384 bytes, SHA256 07e33eb2c6aab5727fbb72c0b403bc83ca8e5695d379106af089b27baf995b25. Upcast FP32 with a NEW Adafactor; the unavailable ancestor FP32/AdamW/RNG is not reconstructed. All100028328 parameters remain trainable; no architecture or tokenizer change.

Hypothesis: fixed-budget full-parameter continuation on bilingual natural-language descriptions and executable Python targets improves free functional code synthesis and both-correct semantic counterfactual pairs over the same frozen L100EC ancestor. One trained model and frozen-parent comparison; combined data recipe, not a causal architecture-ablation study. Counterfactual instructions test semantic sensitivity. No external teacher, manual inference solver, constrained decoding, retries or model-output repair.

Fixed1024 updates, batch4, seed111061, sampler111062. Adafactor peakLR.006,32-step warmup, cosine floor.15, clip1, foreachFalse. FP32 weights/gradients, BF16 matrix operations, FP32 delta state, activation checkpointing. Four local CPU threads. Checkpoint model/optimizer/RNG every256 updates; final1024 only, no held checkpoint selection. Three throughput-probe updates were discarded before the real start.

4096training prompts/2048semantic groups, balanced English/Russian;96validation prompts. Fresh experiment-held IID160,surface160,length80,counterfactual160/80pairs. Families: two-step integer transformations, piecewise conditional transforms, list filter+map and filtered aggregation. Conditions contain words and numeric constants, not target Python operators/statements. Group exact affine/piecewise or pointwise list behaviour over the full registered breakpoint domain BEFORE rendering; bilingual duplicates only within splits. No claim that unavailable EC original data are fully decontaminated.

Inference: prompt text only, full8192vocabulary, greedy96tokens, EOS required. Raw code is syntax-checked and executed in a separate resource-bounded allowlisted Python worker on negative/boundary/empty-list tests. It is a constrained synthetic code benchmark, not EvalPlus/HumanEval or universal intelligence.

Primary gates: IIDpass@1 gain>=15pp, surface>=10pp, length>=5pp, both-correct counterfactual gain>=10pp versus parent. These are mechanism gates, NEVER a Qwen-parity declaration. Prior Qwen reference scores are on different fixtures.

Preregistered JSON SHA256:98c5020f88ef329557a94313bd54c4360bc608cb2f1f768a48132b281c5f119d. Training data SHA256:efba4d6cdc11abbfe88155c5b765dd19802e7b46af6048f117229d5dfcce9c2b. Exact source/data and raw metrics will be preserved.12engineering tests passed before training. No active CI/local training was observed at startup. L100QL/L100LB/L100V/L100CS and other parallel experiments are NOT restarted or overwritten.

Publication scope of this commit: protocol and hashes only; weight bytes have NOT been uploaded. A manifest does not count as a model release.
