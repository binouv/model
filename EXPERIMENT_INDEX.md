# FlyGraph Experiment Index

| Round | Topic | Status | Canonical artifact |
|---|---|---|---|
| R3–R18 | early multimodal/sparse scaling | historical | `reports/archive_r3_r14/`, `reports/RESEARCH_REPORT_R18_RU.md` |
| R17–R20 | fixed-compute scaling + robust audio | historical/recovered | `history/r17_r20/` |
| R21–R22 | downstream audio routing | historical/recovered | `history/r21_r22/` |
| R23–R26 | cognition-first long-hop memory | **old absolute long-hop metrics invalidated** | `reports/CORRECTION_NOTICE_R23_R26.md` |
| R27–R34 | corrected benchmark, reusable operators, noisy memory, multi-hypothesis retrieval | completed | `runs/r27/` … `runs/r34/` |
| R35–R44 | consistency/value verification and recovery triggers | completed; mixed/refuted | `runs/r35/` … `runs/r44/` |
| R45 | candidate-level branch value | confirmed for 128-hop retention on its held set | `runs/r45/` |
| R46–R51 | 256-hop stress + retrieval/ranking diagnostics | completed | `runs/r46/` … `runs/r51/` |
| R52 | learned protected ECC address geometry | completed | `runs/r52/` |
| R53–R54 | multi-hypothesis ECC decoding + adaptive ECC branching | completed | `runs/r53/`, `runs/r54/` |
| R55–R57 | protected branch reservation / address diversity / joint-state diversity | **refuted** as current causal bottleneck | `runs/r55/` … `runs/r57/` |
| R58 | shallow future-value reranking at prune boundary | **CONFIRMED** on predefined criterion | `runs/r58/` |

**Current conclusion:** candidate support is usually sufficient on the tested 64/128-hop regime; immediate prune-boundary value is a causal bottleneck. R58 H=1 lookahead improves held long-horizon accuracy but costs ~2.45x edge evaluations at 128 hops.

**Next:** R59 distill/cache the R58 one-step value signal from current candidate/ancestry features, with exact R58 as the single ablation/reference. 100M/200M/300M scaling remains deferred.
