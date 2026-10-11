# L100QL source delivery

Restore the exact locally tested source with `python runs/l100ql_20261011/restore.py`.

Eight binary source parts concatenate into a 19,824-byte XZ JSON mapping containing 13 source/config/engineering-report files, SHA256 `3ef7db1c7bcb6546dbfa432230981ba791f6a684bb4e6604928d8d123009c5f7`. Every uploaded part's returned Git blob SHA matches its local bytes. A clean local restore reproduced all 13 files exactly. This is source publication, not a model weight release.

Contents include the original100M hybrid backbone, learned query lexical-pointer model, generator, matched full-parameter trainer with atomic native checkpoints and resume, prompt-only evaluation, inference,22 tests and fixed preregistration. Training and fixed evaluation data plus tokenizer are separately packaged; no data or large weight bytes are hidden inside these source parts.

At this commit the primary lexical arm has completed all512 optimizer updates and all6 held evaluation splits. The matched control is still being trained locally in the active response. No completed paired scientific result, statistical superiority, or Qwen parity is claimed until aggregation. Native weights and optimizer are saved locally but are NOT uploaded in this commit.

Parallel L100V/L100CS and all earlier experiment directories are unchanged. No remote own-model training, paid service, or user GPU was used.
