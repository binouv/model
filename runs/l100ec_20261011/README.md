# L100EC — recovered prior results and publication audit

Publication audited 2026-10-11. This directory adds available prior summary metrics and model metadata without modifying other experiments. No new training or held evaluation occurred during this audit.

The mounted prior result reports two completed 100,028,328-parameter models, curriculum and shuffled-order control, 1536 updates each. Saved IID correct counts are47/240 and37/240; surface15/120 and11/120; length0/120 for both; counterfactual42/120 and28/120, with19/60 and13/60 both-correct pairs. The registered hypothesis was NOT_CONFIRMED_IN_REGISTERED_PROTOCOL. No Qwen superiority claim.

## Actual files versus historical manifests

Both BF16 exports are physically present and were SHA256-hashed again. Each has200069384bytes. Curriculum:07e33eb2c6aab5727fbb72c0b403bc83ca8e5695d379106af089b27baf995b25. Shuffled:2047dc31ccb58c9326fca9eeffff78c01bb323aec9462b7622d63738a2cb63e1.

The FP32 weights and AdamW/RNG files named in earlier checkpoint manifests are NOT mounted in this current context. Original L100EC training/inference sources, full training data and all primary raw predictions are also not mounted. This is a recovery/publication limitation, not proof that the original training was never completed or that copies cannot exist elsewhere.

The local downloadable ZIP packages actual available BF16 weights, tokenizer, model configuration and prior metrics/report. It is NOT an exact-resume package, a complete training reproduction, or a GitHub weight release. Its metadata is in archives/DOWNLOAD_MANIFEST.json. No model inference was performed anew by this audit.

L100CS has only its preserved preregistration available here; no completed checkpoint is established. The config records earlier intentions, not fresh execution evidence. Existing experimental results and parallel branches are untouched.

**Publication is partial: summaries, configuration and audit are in Git; model bytes, original optimizer/RNG and the full historical source/data collection are not all uploaded.**
