# Token-Delta: latent semantic planning and compact textual realization
Status: **2 completed CPU-only synthetic experiments, not a Qwen performance claim**.
This track is independent of FlyGraph cognition R-series and the lossless RAM/VRAM compression track.

## Read
- [Preregistered hypotheses](PREREGISTRATION_2026-10-12.md)
- [Full Russian scientific article](SCIENTIFIC_ARTICLE_RU.md)
- [Eight-token future-state experiment](src/tiny_future.py)
- [Single-latent / two-style trained experiment](src/latent_answer.py)
- [Tests incl. checkpoint reload](src/test_latent_answer.py)
- [Probe metrics](results/metrics.json)
- [Style-transfer metrics + causal ablations](results/latent_answer_metrics.json)
- [Predictions on heldout combinations](results/examples.json)

## Findings
Motif 8-token probe mean 98.07%, iid-random control mean 6.20% (chance=6.25%).
Planner 4,736 parameters + renderer 31,249; same 32D latent -> short/detailed texts.
Paired heldout 102/102; zero/shuffled plan 0/102.
Not pretrained language understanding. No 27B or 20K context speed measurement.

## Reproduce
Run in a trusted environment with Python 3.12/3.13, PyTorch CPU and NumPy:

```bash
python -m pip install torch numpy
python research/token_delta/src/tiny_future.py --train-steps 300 --probe-steps 230 --batch 128 --eval-samples 1024 --out research/token_delta/results
python research/token_delta/src/latent_answer.py --steps 600 --batch 64 --out research/token_delta/results
cd research/token_delta/src
TOKEN_DELTA_CKPT=../results/latent_answer_training_checkpoint.pt python -m unittest test_latent_answer -v
```

The sandbox experiment saved real torch checkpoints with weights + optimizers + RNG. Check remote commit content for actual checkpoint bytes; a SHA manifest alone is not binary publication. PyTorch pickle checkpoints must be loaded only from trusted sources.


## Pretrained-teacher continuation (R02–R03, 2026-10-12)

- **[R02: Qwen3-0.6B + naive 48D answer plan](r02_qwen_teacher/REPORT_RU.md)** — real frozen pretrained teacher (official pinned checkpoint), identical supervised dataset/baselines; **negative result**: 0/48 exact Qwen-student vs 48/48 hash-BOW control on heldout question triples. Original frozen Qwen features, both actual trained student checkpoints, metrics and all predictions are checked in.
- **[R03: supervised semantic bottleneck](r03_semantic_bottleneck/REPORT_RU.md)** — same SHA-verified Qwen hidden states; linear probe extracts 33/48 new-synonym semantic triples vs 4/48 BOW; a 172,754-parameter student generates both styles correctly on **30/48** synonym holdouts vs **5/48** BOW. Exact pretrained-data provenance, all probe/joint-student checkpoints, optimizer state, metrics, predictions and SHA manifests checked in.
- **[Combined full scientific article in Russian](SCIENTIFIC_ARTICLE_RU.md)** — R01+R02+R03, including negative and positive findings and limitations.
- **[Offline demo of real Qwen-derived semantic plan](r03_semantic_bottleneck/src/demo_inference.py)** — produces two text styles from one 48D plan without reloading or rerunning Qwen, using the stored pretrained-derived features and the published R03 checkpoint.

These small models do **not** faithfully reproduce a 0.6B model's entire answer distribution; the task is explicit three-slot English fact copying with unseen combinations/synonyms, not general reasoning. Do not claim end-to-end speedups, 20K context, CUDA performance or lossless weight/KV compression from these results.
