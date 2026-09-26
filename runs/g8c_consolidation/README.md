# FlyGraph G8C — six completed checkpoint continuations

**Completed. The registered >=5 percentage-point IID gain criterion was NOT met.**

A narrow positive effect exists: trace-to-final averaged 5.42% versus 0.75% for final-to-final. Both-correct counterfactual pairs remain 0/200 for every seed. These are **469,648-parameter English synthetic generative probes**, not new 300–800M models or Qwen parity.

Actual G5S CPU-release model, AdamW, torch/Python/sampler RNG states were restored. Each of six parents continued for 1,000 updates. No parent was retrained. The three ancestral seeds are 7601/7602/7603. All 1,600 fresh held questions are disjoint from the available parent canonical groups. Second-stage inputs and operations match; full two-stage cost does not, because parent trace training used more tokens.

| Held | Trace-to-final mean | Final-to-final mean |
|---|---:|---:|
| IID, 400 questions | 5.42% | 0.75% |
| Unseen surface, 400 | 2.67% | 0.33% |
| Extrapolation, 400 | 0.67% | 0.17% |
| Counterfactual questions, 400 | 5.17% | 0.42% |
| Both-correct pairs, 200 | 0% | 0% |

IID raw counts: 19/24/22 versus 2/3/4 out of 400. The mean gain 4.6667 points remains below the preregistered 5-point threshold. The threshold was not changed. Questions repeat across seeds; they are not 1,200 independent IID cases. No gold or executor enters model inference.

## Real published bytes

https://github.com/binouv/model/releases/tag/flygraph-g8c-consolidation-36238890991

Prerelease 397210425 contains six safetensors files, six configurations, 24 native model/optimizer/RNG checkpoints, six raw JSON.gz files, source/data/results and a manifest: 44 assets. All 43 payloads were downloaded back in CI and SHA256-verified. The final 25 payloads were also delivered into the conversation container: 19,200 before/after predictions rescored, 96 reloaded generations matched, and all six optimizer/RNG next-update checks passed. Diagnostic updates were discarded.

See `metrics/FINAL.json`, `metrics/FINAL.csv`, `reports/RESEARCH_REPORT_RU.md`, `reports/PUBLICATION_STATUS.json` and `reports/NEXT_STATE.json`. Historical progress snapshots do not supersede the completed result. Registered source and configuration are unchanged.

## Reproduction

The release `SOURCE_DATA_RESULTS.tar.gz` preserves exact inputs, raw results and original code. Native `.resume.pt` files contain actual model/optimizer/RNG state; only load them after authoritative SHA256 verification. Git source directories do not themselves contain these checkpoint bytes.

The conversation's full local ZIP includes the six final native checkpoints and safetensors, a CPU inference CLI and `src/verify_bundle.py`. Run that verifier from the extracted complete bundle; it checks all files, rescoring and checkpoint replay. Twelve tests passed from clean source extraction. CPU inference does not embed an answer solver.

The current gain is concentrated in list and memory questions. IID arithmetic and code traces remain zero for the trace-parent arm. Changing the queried memory key changed predictions on only 4/1/5 of 100 pairs. Query-grounded contrastive supervision is proposed next, not yet tested. G8C held data must not become training data. Broad RU/EN/code competence and the 300–800M goal remain unachieved.
