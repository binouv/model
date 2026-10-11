# L100V — completed recovery and engineering preflight

Date: 2026-10-11. This is an engineering-stage receipt, NOT a completed training result or a Qwen superiority claim.

Recovered real L100EC BF16 export A from the user's Library. SHA256: `07e33eb2c6aab5727fbb72c0b403bc83ca8e5695d379106af089b27baf995b25`; bytes: 200069384. Strictly loaded into the compatible existing FlyGraph L100M architecture: 100028328 parameters, 135 state tensors, all finite. Config SHA256: `2015d526e5d61220366a61411684db81ca8638b8fbef72cb0202f3c8f2ea6478`; tokenizer SHA256: `aa6f3b462d4d1bb9ace09d0421a1879500fea91127a347a29df3e46a6f58c500`.

Native EC optimizer/RNG and L100CS checkpoints are not available in this runtime. The new run is explicitly a BF16-to-FP32 warm start with a NEW optimizer; no claim to resume the missing CS updates. Later R/S report records were read and are not overwritten or rerun.

The new verified-skills data/trainer/evaluator passed 28 actual local pytest tests. Coverage: group disjointness, arithmetic symmetry, memory history/query grouping, affine-code equivalence, loss masks, weighted loss, cache causality, activation-checkpoint gradient equivalence on a small engineering fixture, reversible tokenizer, and strict generated-code AST rejection. These small-fixture tests do not substitute for a completed100M training or benchmark.

L100V preregisters one finite combined recipe: 2048 full-parameter updates, first512 elementary then1536 composed, batch8, AdamW lr0.0003, semantic computed-token weight8. The architecture is unchanged. New data includes arithmetic, conditional execution, code tracing, list sums, mutable memory, and Python functions requested in natural RU/EN language. A static parent comparison measures total training gain, not an isolated causal effect of each ingredient. Main held suite: 240 IID,120 surface,120 OOD,160 counterfactual questions/80 memory pairs. Exact EC train groups are unavailable, so full decontamination against that missing corpus is not claimed.

Local runtime: torch2.10.0+cpu, no CUDA device, 4 GiB cgroup RAM. The user's RTX3090 is not being used. No own-model training is delegated to remote CI. Source and completed metrics will be published as separate verified stages. Weight bytes are NOT contained in this receipt.
