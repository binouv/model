# Reproduce or resume G3L

Use the completed G3SourceData and actual step2000 model+AdamW+RNG checkpoints for full/hybrid seeds7301/7302/7303. Copy G3data byte-for-byte to this directory/data; copy data/manifest.json too. The trainer verifies all dataset hashes against the parent checkpoint before optimization.

For each seed/variant run `python src/continue_train.py --variant full --seed 7301 --parent /path/to/G3/checkpoints/full_s7301/step2000.pt`. Add `--resume /path/to/G3L/checkpoints/full_s7301/step4000.pt` to resume an actual saved boundary. Unsaved log tails must be explicitly quarantined first, not silently included. After all six final8000checkpoints exist, evaluate with --resume pointing to step8000 plus --eval-only. No final checkpoint selection on held results.

A G3safetensors inference release does NOT include AdamWmoments/RNG and cannot stand in for these exact continuation parents. SourceData bundles preserve the full parent checkpoints separately. Do not relabel a new-optimizer warm start as exact resume.
