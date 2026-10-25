# Reproduction

Run from the repository root with Python3.12 and `requirements.lock.txt` dependencies. This directory contains actual public records, frozen features and native float64 probe coefficients. Full FlyGraph checkpoints were unchanged; they remain in the ancestor EXT2/EXT3 ordinary Git binary parts, not duplicated here.

Reconstruct the frozen models with the existing lossless utility:

```sh
python runs/ext3_20261025/src/checkpoint_parts.py restore runs/ext2_20261022/artifacts/selected_model_parts runs/ext2_20261022/checkpoints/step_3584/model.safetensors
python runs/ext3_20261025/src/checkpoint_parts.py restore runs/ext3_20261025/artifacts/selected_model_parts runs/ext3_20261025/checkpoints/step_0640/model.safetensors
```

The exact expected full SHA256 values are in `config.json`; the loader checks them. Preserve the parent directories and tokenizer. The historical checkpoint metadata is pinned in `PARENT_CHECKPOINTS.json`. No public model/API download is required to replay our saved artifacts.

Verify already saved data, source identities, native float64 refit and288 dev/test readout predictions:

```sh
python runs/diagnostics4_20261025/src/verify_artifacts.py --historical
```

Omit `--historical` for a standalone run directory; local train/dev/test disjointness and record hashes are still checked, while prior split audit needs the ancestor run data. Historical unseen pretraining contamination cannot be ruled out.

To reproduce full neural inference, copy this directory into a fresh sibling directory under `runs/` and remove only generated `metrics/`, `artifacts/` and `logs/` contents in that disposable copy; preserve `data/`, `src/`, configs and registrations. `common.ROOT` resolves to the script's containing run, while model paths remain sibling EXT2/EXT3 references. Output JSONL uses exclusive creation and intentionally refuses to overwrite existing evaluation results. Then:

```sh
python runs/NEW_DISPOSABLE_COPY/src/run_diagnostics.py
python runs/NEW_DISPOSABLE_COPY/src/analyze_results.py
python runs/NEW_DISPOSABLE_COPY/src/verify_artifacts.py --historical
python runs/NEW_DISPOSABLE_COPY/src/architecture_audit.py
python runs/NEW_DISPOSABLE_COPY/src/verify_chunked_delta.py
```

Data selection seed2026102504; inherited deterministic runtime seed20261025, as recorded in environment.json. Data source archive/member hashes and split rules are in `data/manifest.json` and `src/prepare_data.py`; preparation requires the original cached OpenCompass archive and ancestor train/dev/test records and must run into an empty data destination. Existing sampled records are sufficient for model/probe replay. Greedy generation uses16 new tokens. Candidate log likelihood uses complete original answers, bounded by context2048; token boundaries/lengths are checked and recorded. Different scoring rules are diagnostic outputs, not interchangeable benchmark wins.

`chunked_delta.py` is optional and never modifies original model.py. To use it in a new inference process after loading these sources:

```python
import model
from chunked_delta import chunked_delta_rule
model.delta_rule = chunked_delta_rule
```

It preserves parameter names/shapes and FP32 recurrent state. Floating-point reassociation is expected; the exact native training/replay path remains the original implementation. Engineering checks use source dev sequences, not generated benchmark questions. The architecture plan describes future experiments, not changes silently applied to saved weights.
