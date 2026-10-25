# FlyGraph EXT3 — counterbalanced answer choices

This bounded experiment continues the existing 100,028,328-parameter EXT2
checkpoint3584. It trains13,391,640 parameters in the last two blocks and final
norm, using published OpenBookQA/PIQA choice permutations and an Alpaca anchor.
No questions, alternatives or gold answers were invented. See REGISTRATION.md,
data/manifest.json and REPORT_RU.md for the registered procedure and actual result.

The original EXT256 and EXT2 are evaluated on the same new external holdouts as
EXT3. Exact Qwen3.5-0.8B and MiniCPM-V-4.7-1B remain unmeasured because official
HuggingFace CONNECT returns403. No general superiority is established.

## Restore native bytes

From this directory, with standard-library Python:

```bash
python src/checkpoint_parts.py restore artifacts/final_model_parts checkpoints/restored_final/model.safetensors
python src/checkpoint_parts.py restore artifacts/final_state_parts checkpoints/restored_final/training_state.pt
```

If dev selected an earlier checkpoint, use selected_model_parts and
selected_state_parts instead. If selected step0, the input is the previously
published EXT2 selected_model_parts, identified in artifacts/MANIFEST.json.
Every raw fragment is strictly below90,000,000 bytes; reassemble before loading.
Model and AdamW tensors retain native FP32 precision. Frozen parameters are
verified bitwise unchanged. Parent/original bytes remain in the previous EXT2
artifacts within this repository, with exact SHA-256 recorded in config.json.

## Offline inference and exact native continuation fixture

Use Python3.12 and requirements.lock.txt (PyTorch2.14.1+cpu).

```bash
python src/infer.py --weights checkpoints/restored_final/model.safetensors --prompt 'User: YOUR_PUBLISHED_QUESTION
Assistant:' --max-new-tokens 16
python src/verify_remote_checkpoint.py --weights checkpoints/restored_final/model.safetensors --state checkpoints/restored_final/training_state.pt --output native_resume_check.json
```

The second command replays one already-published external training update twice
from separate native loads and compares exact parameters. It does not modify
saved checkpoints or count as an additional quality run.

## Reproduce locally

With the previous runs/ext2_20261022/artifacts present in this checkout:

```bash
python src/new_reproduction.py /workspace/flygraph-ext3-reproduction --minutes 70
```

This creates an isolated destination, restores both comparison inputs, reruns
initial dev/training, exact step64→128 replay, frozen three-model evaluation,
report and local binary packaging. --prepare-only creates inputs without training.
No publication is performed. Reproduction needs the same software/CPU stack for
bitwise equivalence; it is not asserted across arbitrary hardware.

Selected source records, immutable dataset identities, original IDs, licenses,
seeds, scripts, loss and raw outputs are included. Original-source selection is
recorded in src/prepare_data.py. The small report/code/data ZIP excludes weights;
actual native bytes are in the adjacent parts. PUBLICATION.json separately
records whether a remote upload/readback was completed. Dataset restrictions,
including Alpaca noncommercial research use, are retained; see DATA_LICENSES.md.
