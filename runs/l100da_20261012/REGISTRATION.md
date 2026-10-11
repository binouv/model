# L100DA — direct-answer mixed-task continuation

Registered before any new task training or held model evaluation. This is not a completed scientific result. Durable weight policy: FLYGRAPH_RUN_POLICY.md, commit 0e691af5f7cc0b23d1b4d86f25368eb988c71d8e.

## Parent and training

- Parent: actual L100EC curriculum BF16 bytes, SHA256 `07e33eb2c6aab5727fbb72c0b403bc83ca8e5695d379106af089b27baf995b25`.
- Weight-only warm start with new AdamW, NOT recovered EC/L100QL optimizer. Lost L100QL is not claimed recovered. Existing L100V/L100CS/L100QL paths remain unchanged.
- Same 100,028,328-parameter architecture and original 8192-token reversible tokenizer. All backbone parameters trainable, no frozen facade, LoRA, inference parser or external solver.
- Exactly 256 optimizer updates; effective batch 8, four length-sorted microbatches of 2. Gradient accumulation normalized by the total supervised token count of the complete batch. No checkpoint during pending accumulation.
- AdamW lr 0.00015, betas (0.9,0.95), weight decay 0.01, foreach false; gradient norm clip 1. 16-step warmup then cosine to 25% of peak at step256.
- First64 updates elementary; afterwards 25% elementary/75% larger-number training, balanced across arithmetic, conditional, code_trace, list_reasoning, memory_update and code_generation. RU/EN represented independently within each family.
- Target is direct `F=<integer>` or Python function, not repeated teacher-forced traces. Offline targets checked independently. No generated code executed unrestricted: code scoring uses a bounded arithmetic AST interpreter.
- Seeds: initialization1210111, sampler1210112; data1210101..1210106. FP32 model/AdamW, BF16 compute, original FP32 delta operator, activation checkpointing. Local CPU four threads; no user GPU, remote own-model training or paid API.
- Local checkpoint every64 completed updates plus final, saving full model/optimizer/RNG/sampler/config identities. Native and separate BF16 export preserved in binary parts <=90,000,000 bytes (default80,000,000), with independent readback required for publication status.

## Frozen data and endpoint

- Train8192 examples,6801 canonical groups. Available L100M held groups excluded from training (850 groups). New held excludes all17234 available L100M canonical groups and current train groups. Arithmetic permutations/operators and memory query variants are grouped; code-generation programs are grouped by equivalent affine mapping.
- Data hashes: train `596e5ba5f63ea716c6bf37122c34d10ac64c13b4b5f4b57206efe63bea90e88d`; validation `409315bbe5048805c357b1e6f102347b28f114d8b725d127afe06ce9d1b94136`; IID `d2e9a23a4310d539764c4221fd65c62364e1bc5a32e941897e263f2c905be3c9`; surface `4ccdbc498594780fec05b046e6e957df725aa2c2c3f1f3669a6a48cb0f4b43cb`; extrapolation `808ded33794464a5258fda0db4ab1212ad6f41352162024d81aa042ea12c553e`; counterfactual `5a37f854196e133210b7bb11e3669e9be1d9f38d780707693b35a9d02d8c3aae`.
- Training max observed210 tokens (hard allowance384), no truncated/rejected long Russian examples. Inference greedy full8192 vocabulary, up to96 new tokens, EOS/length and raw outputs retained for both parent and new model.
- Evaluation: IID96, new surface48, larger numbers/longer computations48, memory32 questions in16 complete query-change pairs. Fixed final step256, no held-based model selection or hyperparameter search.
- Primary gate: numeric IID before/after gain >=5pp, paired bootstrap95% lower bound >0, and counterfactual both-correct pairs not lower than parent. Bootstrap10000 samples, seed1210113.

## Scope and engineering checks

One continuation pilot with before/after comparison, not a matched-control causal conclusion. Twenty local engineering tests passed: independent targets, loss masks/padding, group disjointness, tokenizer reversibility, hashes, counterfactual pair integrity, restricted code evaluator, prompt-only inference path, and exact next-step model/AdamW/RNG replay on a small architecture fixture. No performance result follows from those tests.

No new Qwen reference inference is included. Six synthetic task families and one seed are not universal language competence. Original EC full training data remains unavailable: absence of inherited EC contamination is NOT proven. Overall goal remains broad superiority to Qwen3.5-0.8B and is not achieved by this registration.
