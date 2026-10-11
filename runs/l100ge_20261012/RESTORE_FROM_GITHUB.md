# Restore L100GE from verified binary Git objects

From a full checkout of `binouv/model`:

```bash
python runs/l100ge_20261012/restore.py
python runs/l100ge_20261012/src/assemble_parent.py
cd runs/l100ge_20261012
python -m pip install torch safetensors pytest
pytest -q tests
python src/infer.py --arm readout_lora --prompt 'What is 2 + 3? Answer:' --max-new-tokens 24
```

`restore.py` losslessly reassembles `source_data_results.tar.xz` (2,510,816 bytes, SHA256 `8366eb33fe83ece5729848e0271e0e3035b4b0e14cbc841b476c37c06bffc422`) from 6 actual Git parts, and `all_native_checkpoints.tar.xz` (572,560 bytes, SHA256 `fbc550be5b3e7777c1c9232f11d42f7eb8189ea38359acfbb6727dcda98a6972`) from 2 actual Git parts; verifies individual full source/data SHA and 64/128/192/256 checkpoint weight/optimizer hashes. `assemble_parent.py` reads the previously uploaded original 100M BF16 weight parts from `models/l100ec_curriculum_bf16_07e33eb2/` and validates SHA256. Does not download/retrain Qwen.

**Result:** LoRA 1/160 versus parent 3/160 held; negative. This is a real adapter to the 100M generative model, but NOT full 100M fine-tuning nor universal intelligence improvement.
