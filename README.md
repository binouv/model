# FlyGraph Research

Canonical source of truth for FlyGraph cognition, reasoning, memory, multimodal experiments, scripts, tests, metrics, reports, and checkpoint manifests.

## Current state
- Latest completed cycle: **R38**.
- Priority: **reasoning + memory**; audio/video are secondary unless they block cognition benchmarks.
- Protected/discrete address state remains separate from reasoning workspace.
- Structured reusable operation primitives remain preferred over generic soft recurrent composition.
- Multi-hypothesis retrieval + consistency verification is the active memory line.
- Historical R23–R26 deep-hop absolute metrics are invalidated by the overwrite bug; see `reports/CORRECTION_NOTICE_R23_R26.md`.
- R38 result is **REFUTED/MIXED**: gated off-policy residual improves 64-hop ranking but regresses 128-hop versus protected cumulative R35 scoring.
- Therefore 100M/200M/300M workspace scaling remains deferred until 128-hop trajectory ranking is stabilized.

## Where to start
- `EXPERIMENT_INDEX.md`
- `reports/RESEARCH_REPORT_R38_RU.md`
- `reports/CORRECTION_NOTICE_R23_R26.md`
- `history/FLYGRAPH_TEXT_HISTORY_R3_R38.txt` — complete available text history bundle.
- `models/FLYGRAPH_SMALL_MODELS_BASE64.json` — lossless small-model artifacts + manifests for larger checkpoints.

All new FlyGraph cycles should read this repository first and commit code/config/tests/metrics/report after completed runs.
