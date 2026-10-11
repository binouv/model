# L100QL — query lexical bypass, registered before training

Engineering checks completed locally: 22 tests passed, 0 failed. This registration is NOT a completed training result and makes no Qwen superiority claim.

Parent: real L100EC curriculum BF16 checkpoint, SHA256 `07e33eb2c6aab5727fbb72c0b403bc83ca8e5695d379106af089b27baf995b25`, verified by reading the local bytes. New AdamW warm start; no claim to restore unavailable EC optimizer. Stored parameters per arm: 100177331; inherited backbone: 100028328. Full backbone remains trainable. The matched control has zero task gradients in the lexical-only projection/kernel/scale, explicitly not counted as active learned control capacity.

Hypothesis: directly routing learned query-token pooling into a local lexical pointer-key bank improves freely generated counterfactual memory answers vs the same query/source/gate-supervised pointer with the lexical score multiplied by zero. Numeric/sign-only copy channel plus full8192-token vocabulary channel in both arms. Inference accepts prompt token IDs only: no parser, latest-key solver, gold, query labels, source positions or answer candidate list.

Two arms: lexical and control. Exactly512 updates each, batch4. AdamW betas(.9,.95), weight_decay.01, clipping1, foreachFalse; body LR.0001, head LR.001;16-step warmup, cosine floor.25. FP32 parameters/optimizer, BF16 compute, original FP32 delta operator, activation checkpointing in both arms, four local CPU threads. No user GPU, remote own-model training, or paid API.

Seed1101271 for initialization; sampler1101273. Four out of five minibatches contain two fresh memory query pairs; every fifth contains four old L100M non-memory final-answer examples. Same ordered batches in both arms. Source/query/gate auxiliary weights .5/.5/.1 in both arms. Fixed final512 endpoint; no held-based checkpoint or hyperparameter selection.

Data:4096 fresh train banks (8192 questions),64 validation banks. Held:200 IID pairs,100 paraphrase pairs,100 longer-bank pairs with unchanged value range,100 renamed-key pairs,100 four-digit-value pairs, plus120 historical regression questions. All available L100M/L100A bank groups excluded; decontamination against unavailable EC original training data is NOT established. Held query/source labels are not stored or sent to inference.

Primary gate: IID both-correct pair gain >=5 percentage points, paired bootstrap95% lower bound >0 (10000 samples, seed1101274), historical regression loss <=2 percentage points. One primary hypothesis and one matched ablation; one ancestral initialization, not multiseed replication. New Qwen run is not part of this finite mechanism experiment; broad goal remains unachieved.

Parallel L100V, L100CS and other experiment directories are not modified, restarted, or claimed completed by this work. Source, raw outputs and model artifacts will be separately identified in completion receipts.
