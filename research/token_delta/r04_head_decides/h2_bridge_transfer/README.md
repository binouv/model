# R04-H2 — Frozen Qwen3.5-4B head → bridge → frozen Qwen3.5-0.8B writer

**Status:** preregistered, original R04 H2 gates unchanged; **12 CPU tests pass**; no pretrained RTX3090 H2 accuracy/speed results yet. See [EXECUTION_LOCK.json](EXECUTION_LOCK.json) for exact model revisions, optimization, split and thresholds. **H1 (~16.7% frozen-head accuracy), H1b and H1c remain negative/inconclusive as originally reported**, not reopened.

## Scientific contract

- Qwen3.5-4B is **frozen**: one prompt-only forward produces final 2560D state. Separately we capture all 1,700 greedy 4B text outputs for training supervision/quality metrics; no oracle enters Qwen inputs. (The 4B answers are known to be frequently wrong; that is the scientific point.)
- Only the `LatentBridge` is trained. The **frozen** 0.8B gets K=4 soft-plan tokens (or K=1/8/16 ablation) and constant assistant-start marker. **No original question, correct answer, teacher answer, length flag or style flag at test inference**.
- Primary H2 trains the writer to reproduce *what the head actually chose*, right or wrong. It separately reports correctness against the programmatic oracle and fidelity to the teacher's text. It must **not** train on 50 teacher-correct examples from the original H1 *test*.
- Input-only hash baseline with identical bridge parameterization; interventions zero/shuffle/negate the latent; writer question-only baseline as a different control; **oracle plan** as an explicitly privileged engineering upper bound.
- Original registered gate: >=75% correct on **all** test, >=80% on teacher-correct, >=15pp against lexical control, >=30pp against shuffled plan, >=2/3 main seeds. These are not adjusted for the known weak teacher.
- End-to-end inference speed and varying reply length are H4/H3, **not claims** from this H2 code. The H2 run is evaluated on deterministic two-digit output targets.

## Structure

```
H2_BRIDGE_TRANSFER/
  README.md
  EXECUTION_LOCK.json
  H2_RUNBOOK.md
  src/h2_dataset.py
  src/h2_extract.py
  src/h2_train.py
  tests/test_h2.py
  data/h2_dataset.jsonl
  data/dataset_manifest.json
  results/                 # GPU artifacts created only after completed runs
  cache/                   # local feature cache excluded from git
```

## Operator sequence — RTX 3090 / Windows PowerShell

Use the same CUDA venv as the validated R04 smoke. **Do not re-install Python or CUDA.** From the `r04_head_decides` directory:

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m unittest discover -s h2_bridge_transfer\tests -v
& $py h2_bridge_transfer\src\h2_dataset.py --out h2_bridge_transfer\data

# Engineering preflight (4 TRAIN samples only; separate directory)
& $py h2_bridge_transfer\src\h2_extract.py --pilot 4 --out h2_bridge_transfer\cache\pilot --batch 2 --shard 4
& $py h2_bridge_transfer\src\h2_train.py --pilot --cache h2_bridge_transfer\cache\pilot --out h2_bridge_transfer\results\pilot

# Scientific teacher targets, no test gold ever enters the teacher
& $py h2_bridge_transfer\src\h2_extract.py --out h2_bridge_transfer\cache --batch 4 --shard 50

# Frozen writer + trainable bridge. Entire predeclared comparison matrix;
# it safely resumes verified completed K/seed runs after interruptions.
& $py h2_bridge_transfer\src\h2_train.py --cache h2_bridge_transfer\cache --out h2_bridge_transfer\results\full
```

The **pilot MUST PASS** before a full GPU run. If `hf` generation disagrees with manual full-prefix greedy decoding, halt the run; inspect `--decode-mode manual` explicitly. We never silently substitute different decoding for favorable scores.

Each fully completed run persists **actual binary checkpoint and AdamW/RNG**, split into <=80,000,000-byte normal Git files when needed, a SHA256 manifest, raw per-item predictions, and a receipt. Keep `cache/` local for GPU feature reproducibility and upload its `feature_manifest.json` plus the exact bytes when publication access is available. `results/full/metrics.json` appears only after all required runs finish.

**Primary report:** all test examples, teacher-correct conditional count, teacher text imitation, truth vs oracle, zero/shuffle/negated controls, hash control, and standalone 0.8B given the question. Do not report quality or speed from the synthetic `pilot`.

## Provenance

H2 fresh dataset is disjoint by `(operator,a,b,c)` from H1 (1700), H1b (128), and H1c (1700). New split 1200/200/300, seed 20261016, canonical SHA256 `f06817743af405c27e6810bd859366d4d1b6c8c6714d02e699ff3f61f1ea65bf`; JSONL file SHA256 `c07f3de8a98e652ab25524a1f822ccbb01ac3b92ef433c6354f1978171131ecf`. This is a **new benchmark**, not H1 test reuse.

Scientific synthesis: [R02–R04 article](../../SYNTHESIS_R02_R04_20261012_RU.md), [original H2 preregistration](PREREGISTRATION.md).
