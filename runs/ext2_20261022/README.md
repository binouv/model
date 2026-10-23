# FlyGraph EXT2, 2026-10-22

This is a new, bounded continuation of the existing 100,028,328-parameter
FlyGraph EXT-step256 model on published Alpaca, OpenBookQA and PIQA records.
The retained tokenizer and model architecture are unchanged. Training uses a
matched User/Assistant format and includes short answer labels. See
`REGISTRATION.md`, `config.json`, `data/manifest.json` and, after evaluation,
`REPORT_RU.md` / `metrics/final_results.json` for the actual outcome.

Qwen/Qwen3.5-0.8B and openbmb/MiniCPM-V-4.7-1B remain unmeasured because the
environment proxy denies their official downloads. No leaderboard values are
substituted. General superiority is not established. PIQA's final test here is
a reserved, disjoint subset of its labeled official validation split; the
official hidden test is not claimed.

## Native checkpoint recovery

`artifacts/MANIFEST.json` identifies the parent, final and selected components.
Each `*_parts/MANIFEST.json` gives ordered raw byte fragments, sizes and SHA-256.
Every part is strictly under 90,000,000 bytes. These are not standalone
safetensors shards. Restore the final checkpoint with Python's standard library:

```bash
python src/checkpoint_parts.py restore artifacts/final_model_parts checkpoints/restored_final/model.safetensors
python src/checkpoint_parts.py restore artifacts/final_state_parts checkpoints/restored_final/training_state.pt
```

If selection chose an earlier checkpoint, restore the corresponding
`selected_model_parts` and `selected_state_parts`. If selection chose step0,
the selected input is `parent_parts`; no new optimizer state for step0 is
invented. Native FP32 model and optimizer bytes are preserved without
quantization. Source weights are separately retained in `parent_parts` to make
the warm start reproducible. See publication receipts for verified remote status.

## Offline inference

Use Python 3.12 and dependencies pinned in `requirements.lock.txt`. PyTorch's
CPU build used here is 2.14.1+cpu. No API, retrieval engine or solver is called.

```bash
python src/infer.py --weights checkpoints/restored_final/model.safetensors --prompt 'User: YOUR_PUBLISHED_QUESTION
Assistant:' --max-new-tokens 16
```

## Fresh reproduction

The existing trainer refuses to overwrite logs/checkpoints. Create an isolated
destination using the included exact selected data and parent byte parts:

```bash
python src/new_reproduction.py /workspace/flygraph-ext2-reproduction --hours 4
```

The command restores the parent, runs one training process, preserves the actual
step128 replay target, verifies step64→128 exact resume, and runs frozen held-out
evaluation after dev selection. `--prepare-only` creates the inputs without
starting compute. The wall-clock cutoff is refreshed; model/data/seed/schedule
are retained. Bitwise reproducibility was verified on this CPU/software stack,
not asserted across arbitrary hardware. No push is performed by reproduction.

`src/prepare_data.py` documents the original selection and expects the prior
immutable source archive/Alpaca file. Selected JSONL records are included so
training reproduction needs no network. Original source IDs and hashes permit
independent auditing against the fixed OpenCompass release and Alpaca revision.

## Files and limitations

Raw per-question predictions, token IDs, likelihoods, training loss, dev curves,
source manifests and recovery verification are under `metrics/` and `logs/`.
The small report/code/data ZIP excludes large native bytes, which remain in the
ordinary repository parts. Intermediate recovery snapshots may be pruned after
a verified successor; hash receipts are retained. Final/selected native bytes
are never replaced by manifests alone.

Research and noncommercial restrictions from Alpaca apply to the trained
checkpoint. Upstream datasets retain their individual licenses; see
`DATA_LICENSES.md`. Unknown historical pretraining contamination and paraphrase
overlap are not claimed absent. Neither lower loss nor valid answer formatting
alone proves stronger reasoning.
