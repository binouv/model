# L100EC recovered runnable BF16 package

This is a publication/recovery task, NOT new training or a new intelligence result.

The actual curriculum and shuffled BF16 checkpoints were rehashed and loaded with the exact inherited L100M model/tokenizer source. Strict state_dict loading succeeded, all100028328parameter values per model are finite, and three generated sequences per arm match after independent reload. Twelve engineering tests pass. One curriculum sanity prompt reaches its48-token cap; this is disclosed in the raw replay, not hidden or scored as a quality benchmark.

The conversation download contains both200069384-byte safetensors files, exact tokenizer bytes, complete inference source, historical available metrics/report and fresh audit. The Git directory contains source and asset hashes; weights are NOT yet Release assets. Copy model/tokenizer files from the downloadable package into the paths in ASSETS.json. Loader rejects mismatched hashes and never loads pickle.

Run from the extracted package:

```bash
python src/infer.py --arm curriculum --prompt-file examples/prompt.txt --max-new-tokens 48
pytest -q tests
python src/audit_runtime.py --arm curriculum
python src/audit_runtime.py --arm shuffled
```

Default computation is CPU, FP32 parameters upcast from BF16 values with BF16 matrix autocast. This conversion does not restore lost FP32 information. No Qwen comparison or optimizer update occurred in this recovery.

Original EC trainer, full training corpus, all primary raw predictions and native FP32/AdamW/RNG bytes are not recovered. Historical manifests refer to earlier files; they do NOT establish present availability. A new optimizer must not be described as an exact resume. L100V and other parallel runs remain untouched.
