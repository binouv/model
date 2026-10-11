# FlyGraph run policy

This file is the durable execution rule for ongoing FlyGraph research.

## Mandatory completion sequence

For every completed FlyGraph cognition run:

1. Execute exactly one primary cognition hypothesis and at most one confirming ablation.
2. Count only fully completed runs/episodes; never present timed-out or partial grids as results.
3. Save local downloadable artifacts:
   - implementation scripts
   - config/calibration files
   - JSON and CSV metrics
   - semantic/tests
   - Markdown research report
   - model/checkpoint artifacts when the run produces them
   - manifest/checksums
4. After local validation, commit/push the completed run to the connected GitHub repository `binouv/model` under `runs/<experiment-id>/`.
5. Push the report and model artifacts as part of the same completed-run publication workflow.
6. Do not wait for user confirmation between hypotheses. Move to the next hypothesis after the completed run is saved and pushed.
7. Keep cognition, reasoning, memory and hypothesis-selection stability ahead of audio/video/body work.
8. Do not resume 200M/300M workspace scaling until memory uncertainty / trajectory selection is substantially stable. Continue the already established 100M line from a verified checkpoint rather than resetting it without an explicit experiment rationale.

If GitHub is temporarily unavailable, preserve the complete local run first and push it at the next available opportunity before treating repository synchronization as complete.

## Mandatory weight preservation and push rule (user instruction, 2026-10-12)

### Actual bytes, not promises

- Every trained model must have its actual weight bytes saved. A manifest, SHA256 string, source archive, Git LFS pointer, or report alone is NOT a saved/published model.
- Store ordinary binary parts in this repository. Hard per-file ceiling: **90,000,000 bytes** (decimal MB, not MiB). Default part size: **80,000,000 bytes**. This applies to weights AND optimizer/training-state parts.
- Split bytes losslessly; do not quantize or discard optimizer precision to meet the size ceiling. An optional BF16 inference export is separate from the exact native training checkpoint.
- Include a versioned manifest with ordered part names, each part's byte count and SHA256, full reconstructed filename/byte count/SHA256, tensor dtypes/counts, model parameter count, parent checkpoint hash, experiment ID and completed optimizer step.
- Do not call byte fragments standalone safetensors shards. Reassemble the original file before loading, using the checked-in `runs/artifact_recovery_20261012/checkpoint_parts.py` (or a tested backward-compatible successor).

### Checkpoint contents and cadence

- Preserve native model tensors, optimizer, scheduler (or exact schedule/config/global step), gradient-scaler state when applicable, Python/NumPy/Torch/CUDA RNG, sampler/data cursor, tokenizer, model/training configs and immutable data/source identities.
- Save after complete optimizer updates (never silently drop gradient accumulation); use temporary paths, flush/fsync and atomic rename. Keep the last verified checkpoint until its successor has loaded and passed validation.
- Set a finite checkpoint interval before training; default every 64 optimizer updates, plus the final update and any controlled stop. Checkpoints are recovery artifacts, not independent scientific results.
- Recovery-only mid-run backups must be labelled `in_progress` or `interrupted`; never claim a partial/timeout run as a completed benchmark. Publish completed experiments promptly before starting the next scientific run. Keep preregistration/source publication separate from completion receipts.
- At the start of every continuation, locate real checkpoint bytes, check hashes, load strictly and inspect actual completed step. Never restart a finished run or fabricate missing optimizer/RNG. Weight-only continuation is explicitly `warm_start`, not `exact_resume`.

### Safe Git publication and independent verification

- Use unique experiment/model directories and normal non-force commits; preserve unrelated and concurrent work. Fetch/check the current branch head before a write and re-read after any conflict.
- Publish code, configs, tokenizer, data or a reproducible immutable data recipe, raw predictions, metrics, report, manifests AND all native model/training-state parts. No credentials or unrelated user files may enter a commit.
- A temporary transfer service is transport only, not the permanent artifact store. Never enable public sharing or expose unrelated private data merely to transfer a checkpoint. Restrict any transport to the exact user-authorized model artifacts and record its status honestly.
- After push, independently fetch/clone the committed files into a clean location, verify every part, reconstruct the full files, verify their SHA256 and load the model. Verify native resume on an engineering fixture and record a fixed-prompt inference receipt. Report the actual commit SHA.
- Distinguish `saved_local`, `uploaded_unverified`, and `published_verified`. A successful metadata commit does not imply successful binary publication. Do not delete the only recoverable bytes while any publication step is pending.
- Provide real downloadable conversation artifacts when available; do not invent sandbox paths or claim remote checks that were not executed.

## Model-quality completion is separate from artifact completion

The target remains a broadly more capable FlyGraph than **Qwen3.5-0.8B**, not merely a successful checkpoint upload or a win on one synthetic task. Track held-out RU/EN language/instruction following, reasoning, code, memory/counterfactual tests, completed reference outputs and inference budget. Mark the overall goal unachieved until sufficient comparable evidence supports it. Missing or truncated Qwen responses are not automatically wrong answers. Preserve negative results and do not select/report only favorable seeds or tasks.

## Publication cadence and portability enforcement (2026-10-12)

1. Checkpoint every **64 successfully completed optimizer updates** by default, plus a final checkpoint. A checkpoint is resumable only when its model tensors, optimizer state, RNG, sampler cursor, configuration, and data identity all pass strict load/replay tests. Never relabel a BF16 inference export as an exact-resume checkpoint.
2. **Periodically push recoverable bytes, not just JSON:** after each validated 64-update checkpoint, when repository write connectivity is available, split and upload the actual binary checkpoint and optimizer/training-state files to a unique `models/<run-id>/step_<N>/` directory. Label these `in_progress` until the prescribed run and benchmark finish. A missing upload is `saved_local`, never `published_verified`.
3. Each ordinary Git blob must be **strictly smaller than 90,000,000 bytes**; normally use 80,000,000-byte pieces. When a provider upload limit is lower, further split into 1–8 MB pieces *without changing/requantizing bytes*. Compute SHA256 for each piece and the concatenation. GitHub LFS pointers, text-only manifests, hash-only files, and empty releases do not count as preserved weights.
4. On publication, perform **clean remote readback** of every binary part, reassemble into a temporary checkpoint, verify each SHA256, load all tensors, verify optimizer/RNG replay on a safe fixture, and record the exact pushed commit SHA. Do not delete the only original training state until this passes.
5. Preserve completed negative experiments and failure states; only publish a research-quality `completed` verdict after the registered evaluation and its control/ablation are complete. Record exact steps, wall time, precision, device, actual sample count, raw generations, external reference coverage and uncertainty. Never infer general superiority from synthetic memory-only results.
6. This is a **standing rule for every future FlyGraph run**, including a continuation invoked from a new chat. If GitHub and the local runtime are disconnected, report the precise status and expose a locally verified bundle, not a fictional push.
