# Checkpoint recovery and parts below 90 MB

This is an engineering recovery operation, not a new training experiment. The L100EC curriculum checkpoint was found, rehashed, split and reconstructed byte-for-byte. L100QL trained primary/control bytes were not recovered from mounted files or available Library title searches; the inspected L100QL Git directory contains registration and source, not trained checkpoints. Existing research directories are untouched.

`checkpoint_parts.py` is standard-library-only Python 3.10+. It implements split, verify, atomic no-clobber restore and non-force Git publication with independent clone/readback. Every part is strictly smaller than 90,000,000 bytes (default 80,000,000). It rejects missing, corrupted, truncated, reordered, duplicate and unsafe parts. Raw-binary Git attributes prevent inherited LFS/text filters from replacing the parts. Existing distinct checkpoints are not overwritten.

19 tests passed locally, including publication to a local bare Git repository, a fresh clone, exact reconstruction, preserving unrelated files and idempotent repeated publication. The local Git test is NOT a successful GitHub binary upload.

```sh
cd runs/artifact_recovery_20261012
python -m unittest -v test_checkpoint_parts
python checkpoint_parts.py split ORIGINAL.safetensors PARTS_DIRECTORY --identity EXPERIMENT_NAME --expected-sha256 EXPECTED_SHA256
python checkpoint_parts.py verify PARTS_DIRECTORY
python checkpoint_parts.py restore PARTS_DIRECTORY RESTORED.safetensors
```

Publication is explicit: use the `publish` subcommand documented under `models/l100ec_curriculum_bf16_07e33eb2/README.md`. A manifest in Git is never evidence that weight bytes were uploaded. The current GitHub commit contains source, tests and metadata only; private Drive and conversation artifacts carry the actual three parts.
