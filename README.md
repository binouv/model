# FlyGraph Research

Canonical source of truth for FlyGraph cognition, reasoning, memory, multimodal experiments, scripts, tests, metrics, reports, and checkpoint manifests.

## Current state
- Latest completed cycle: **R58**.
- Priority: **reasoning + memory**; audio/video remain secondary unless they block cognition benchmarks.
- Protected/discrete address state remains separate from reasoning workspace.
- Frozen R52 ECC + R54 candidate support + R35 trajectory scoring are the current memory stack.
- R57 refuted address/state crowding as the causal bottleneck; its 128-hop diagnostic showed correct continuations were generated but could lose at pruning.
- R58 **CONFIRMED** shallow H=1 future-value reranking on fresh held seeds: 64-hop answer 91.67% vs 83.33%, 128-hop 83.33% vs 66.67%, with no 8-hop regression.
- R58 eliminated observed long-horizon true-path pruning losses on its held set, but costs ~2.45x edge evaluations at 128 hops.
- Therefore 100M/200M/300M workspace scaling remains deferred. Next target is distilling/caching the successful lookahead value at much lower compute.

## Where to start
- `EXPERIMENT_INDEX.md`
- `runs/r58/RESEARCH_REPORT_R58_RU.md`
- `runs/r58/r58_metrics.json`
- `models/r58_shallow_lookahead_policy.json`
- `reports/CORRECTION_NOTICE_R23_R26.md`

All new FlyGraph cycles should read this repository first and commit code/config/tests/metrics/report/model artifacts after completed runs.
