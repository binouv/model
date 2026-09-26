# FlyGraph G8C — six completed checkpoint continuations

**Completed. Registered ≥5pp IID gain criterion NOT met.** Narrow positive effect: trace→final5.42% versus final→final0.75%; both-correct counterfactual pairs0/200 for everyseed. These are469648-parameter English synthetic generative probes, NOT new300-800M systems orQwen parity.

Actual G5S CPU-release model/AdamW/torch/Python/sampler RNG restored. Each of6parents continued1000steps; no duplicate parent training. Three ancestral seeds7601/7602/7603. All1600fresh held questions disjoint from available parent canonical groups. Same second-stage inputs/updates/operations, NOT equal fulltwo-stage cost: parent trace supervision used more tokens.

| Held | Trace→final mean | Final→final mean |
|---|---:|---:|
| IID400 | 5.42% | 0.75% |
| Surface400 | 2.67% | 0.33% |
| Extrapolation400 | 0.67% | 0.17% |
| Counterfactual400 | 5.17% | 0.42% |
| Both-correct pairs200 | 0% | 0% |

IID raw counts:19/24/22 versus2/3/4 out400. The mean gain4.6667pp stays below the preregistered5pp; threshold was not changed. Same400questions reused across3seeds, not1200independent cases. No gold or executor enters model inference.

## Real release

https://github.com/binouv/model/releases/tag/flygraph-g8c-consolidation-36238890991

Published prerelease397210425 contains real6safetensors,6configs,24native model+optimizer/RNG checkpoints,6rawJSON.gz,source/data/results andmanifest. Every43payload asset redownloaded onCI andSHA256-verified. Final25payloads delivered back to the conversation container: all19,200before/after raw predictions rescored,96reloaded generations identical, all6optimizer/RNG next-update replay checks passed. Diagnostic updates were discarded, not new training.

See `metrics/FINAL.json`, `metrics/FINAL.csv`, `reports/RESEARCH_REPORT_RU.md`, `reports/PUBLICATION_STATUS.json` and `reports/NEXT_STATE.json`. Historical progress snapshots are not the current completed state. Original registered configuration and source remain unchanged.

## Replay

Release SOURCE_DATA_RESULTS.tar.gz preserves all inputs, rawresults and code. Native `.resume.pt` files contain actualweights/optimizer/RNG; only load after authoritative SHA256 verification. Source-only Git does NOT itself contain the checkpointbytes. The conversation's full local ZIP includes sixfinal nativecheckpoints/safetensors, aCPU inferenceCLI and `src/verify_bundle.py` for complete re-score/replay. Twelve engineering tests passed from clean source extraction. No complete hardware FLOP measurement was made.

Current gain is concentrated in list/memory tasks; IID arithmetic/code-trace remains0for the trace-parent arm. Changing queried memorykey changes predictions only4/1/5of100pairs. Query-grounded contrastive supervision is proposed next, NOT yet tested. G8held must not become trainingdata. Broader pretrained RU/EN/code and target300-800M remain separate unmet goals.
