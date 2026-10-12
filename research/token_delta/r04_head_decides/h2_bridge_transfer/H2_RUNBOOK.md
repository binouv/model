# H2 GPU acceptance/runbook (RTX3090, Windows)

## Acceptance gate before training

1. Previous R04 smoke passed and 4B/0.8B revisions match `MODEL_LOCK.json`.
2. 12 unit tests pass and full H2 oracle dataset hash equals `EXECUTION_LOCK.json`.
3. Pilot 4 **train-only** samples: pinned 4B head forward, teacher greedy output, correct BF16 hidden shape, no target leakage.
4. Pilot writer: frozen 0.8B, `K=4` bridge learns from teacher-produced targets. Manual causal greedy must match `transformers.generate(inputs_embeds=...)` for identical soft-prefix; if not, stop and use `--decode-mode manual` in a **new** run.
5. No inference claim from pilot and no training using original H1/H1b/H1c test examples.

## Full science

- `src/h2_extract.py` writes paired verified safetensors+JSONL shards, then `cache/h2_head_features.safetensors`, `teacher_outputs.jsonl`, `feature_manifest.json` containing full SHA and full 1700 samples.
- `src/h2_train.py` trains `head` / `hash` K4 × seed 11/22/33, Qwen-head K1/K8/K16 × seed11 and privileged `oracle` K4 × seeds 11/22/33. Same fixed 400 steps, bs4, lr3e-4. No model LoRA. `results/full/run_receipts/*` allows resuming completed model variants with exact SHA checks. The set is complete only after all 12 runs and 300-sample standalone writer baseline.
- Student is evaluated **free-running** on 300 untouched H2 test tasks, without any teacher gold. Raw outputs and all control outputs are stored. The **full H2 status is NOT_CONFIRMED** unless *all* original H2 gates pass for at least two out of three K4 seeds.
- H2a `oracle` source explicitly sees the correct two digits at test and is **upper bound** only. Never mix it into genuine head transfer metrics.

## Storage and security

- Checkpoint <=80,000,000 bytes/part, actual binary bytes plus full SHA/part SHA manifest; trained bridge, optimizer and RNG included.
- If a run dies, preserve existing complete receipts, never label partial interrupted output as a scientific result; rerun exact command to resume fully completed variants.
- With NeoXider 403 against binouv/model, upload the raw result branch to a fork and provide its URL, or attach the original files to chat. Do not invent or recreate SHA/checkpoint objects from summary tables.

## Reporting next

Report `cache/feature_manifest.json` (especially `test_exact` and feature SHA), pilot receipts, `results/full/metrics.json`, `predictions.jsonl`, all `run_receipts` + checkpoint manifest, and `writer_only_question_baseline.jsonl`. Separately record peak VRAM and exact package versions. True head→writer end-to-end latency is H4; **these outputs do not justify an acceleration claim**.
