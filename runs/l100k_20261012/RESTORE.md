# L100K — checkout and restore native checkpoints

From repository root:

```bash
python runs/artifact_recovery_20261012/checkpoint_parts.py verify runs/l100k_20261012/artifacts/all_native_parts
python runs/artifact_recovery_20261012/checkpoint_parts.py restore runs/l100k_20261012/artifacts/all_native_parts ./L100K_All_64step_Native_Checkpoints_20261012.tar.xz
mkdir -p ./l100k_recovered
python -m tarfile -e ./L100K_All_64step_Native_Checkpoints_20261012.tar.xz ./l100k_recovered
python -m tarfile -e ./runs/l100k_20261012/artifacts/source_and_metrics.tar.xz ./l100k_recovered
cd l100k_recovered
python -m pip install torch safetensors pytest
pytest -q tests
python src/train.py --evaluate
```

All native weights and optimizer/RNG training_state files are actual binary Git content, not hash-only pointers. Remote byte-for-byte readback of individual final model/state objects and all three parts passed; see verification/GITHUB_READBACK_RECEIPT.json. Local checkpoint reload replay passed 6 tests, including exact next update. This is a **narrow key-value memory circuit**, not an unrestricted 100M language model.
