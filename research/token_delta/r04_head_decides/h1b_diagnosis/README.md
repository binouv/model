# R04-H1b — Frozen 4B failure diagnosis

**R04 H1 already failed. Keep that result unchanged.** This side experiment answers whether poor 4B performance comes from strict output formatting, a too-small 12-token budget, inability to compute both steps without explicit reasoning, or insufficient one-step skills.

- [Predeclared protocol](PREREGISTRATION.md)
- [Deterministic NEW dataset, no H1 overlaps](src/h1b_dataset.py)
- [GPU benchmark, four inference arms](src/h1b_run.py)
- [CPU-only audit of exact original 300 H1 outputs](src/h1b_audit.py)
- [Unit tests](tests/test_h1b.py)
- [Original H1 negative report (user-reported; binary publication pending)](../h1/H1_RESULT_REPORTED_20261012.md)

Run from `D:\AI\model_r04\research\token_delta\r04_head_decides`:

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m unittest discover -s h1b_diagnosis\tests -v
& $py h1b_diagnosis\src\h1b_dataset.py --out h1b_diagnosis\data

# Reuse user's original H1 results: read-only audit (valid vs formatting errors).
& $py h1b_diagnosis\src\h1b_audit.py `
  --original-data h1\data `
  --original-predictions h1\cache\head_test_predictions.jsonl `
  --out h1b_diagnosis\results\original_audit.json

# Start with 16 new test problems (not original H1 300): 4 arms per case.
& $py h1b_diagnosis\src\h1b_run.py `
  --data h1b_diagnosis\data --out h1b_diagnosis\results\pilot --max-items 16

# Then all 128 new cases. Independent results directory.
& $py h1b_diagnosis\src\h1b_run.py `
  --data h1b_diagnosis\data --out h1b_diagnosis\results\full --max-items 128
```

The benchmark is **resumable** from `diagnosis.partial.jsonl`. Only full arm coverage produces `diagnosis_metrics.json` and `diagnosis_predictions.jsonl`. Use different outputs for pilot/full; never overwrite completed results. The official 4B teacher weights are reused from the same pinned HF revision and should be present in `HF_HOME` cache. No Qwen training happens in H1b.

Important: the original H1 33 checkpoints and ~81MB native bytes are *on the user's Windows machine*. They are **not** accessible to the GitHub connector. The summary published here cannot be substituted for their native artifact upload; they must be pushed by a writable account or via a fork/PR, or explicitly provided as files for synchronized publication. Local commit `c8e3641` should remain protected.