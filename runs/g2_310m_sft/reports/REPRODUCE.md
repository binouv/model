# Exact G2 data and training reproduction

Run from this directory:

```sh
python src/setup_data.py
pytest -q tests/test_protocol.py
python src/run.py --base /path/to/G1/release_bf16 --arm answer_only
python src/run.py --base /path/to/G1/release_bf16 --arm full_sequence
python src/run.py --base /path/to/G1/release_bf16 --arm baseline --baseline-only
```

The first command needs only Python stdlib and the committed base tokenizer implementation. It joins four committed binary XZ fragments, verifies every SHA256, reconstructs the exact 139193-byte tokenizer, then generates and verifies all six JSONL files. All seven file hashes matched in a fresh local reconstruction. No network is needed. The fixed vocabulary was trained on the G1 training corpus, NOT on G2 tests.

The G1 BF16 model weights are a separate prerequisite, not included in this Git directory. Hash-verified local conversation archives contain them. Model index SHA256 must be `430a87eff7986f7b3b0e334696c579ef5a6184bf1d597846a9c675ff28930ebd`. Training starts from these rounded weights and a new optimizer, not from unmounted G1 FP32 state. Both arms train all309,993,253 parameters. Only completed final checkpoints are used for the preregistered final evaluation.

G2 train/test semantic groups are disjoint within this generator. This does not establish semantic disjointness from every older G1 arithmetic example or independence of real-world benchmarks. Numeric answer accuracy is not an internal reasoning-trace proof. These are deliberately limited experimental tasks.
