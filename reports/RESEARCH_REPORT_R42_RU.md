# FlyGraph R42 — multi-step branch-value verifier

**Статус:** REFUTED / MIXED  
**Приоритет:** интеллект / reasoning / память. Audio/video не затрагивались.

## Точная гипотеза

A pairwise verifier supervised by short multi-step protected branch rollout value will identify useful R35-vs-residual divergences better than immediate branch labels, improving noisy 64/128-hop reasoning while preserving short-horizon accuracy.

R41 показал, что immediate binary preference недостаточен. R42 оставляет те же disagreement features и routing policy, но обучает main label по 4-step protected branch rollout value: future true-edge reachability, intermediate state alignment и query→next-subject alignment. Oracle trace применяется только для train labels.

## Протокол

- Train seeds [4211, 4212, 4213]; calibration [4221, 4222]; held [4241, 4242, 4243].
- Noise p=.30, beam=32, expand=8, label rollout=4 steps, local label beam=8.
- Единственная абляция: идентичные features/model/inference, но immediate R41-style training label.

Train comparable disagreement pairs: 190; multi-step vs immediate label flip rate: 12.63%.

## Held-out memory/reasoning

| Hops | N | Greedy | R35 fixed32 | **R42 multi-step value** | Immediate-label ablation | R42 final true path |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24 | 100.00% | 100.00% | **100.00%** | 100.00% | 100.00% |
| 32 | 18 | 50.00% | 83.33% | **83.33%** | 83.33% | 94.44% |
| 64 | 18 | 16.67% | 83.33% | **83.33%** | 83.33% | 83.33% |
| 128 | 12 | 0.00% | 25.00% | **33.33%** | 33.33% | 16.67% |

## Вердикт

**REFUTED / MIXED** — short protected rollout value did not satisfy the full held-out criterion; current disagreement features do not expose enough future value for reliable correction.

8-hop Δ vs greedy +0.00 п.п.; 64-hop Δ vs R35 +0.00; 128-hop Δ vs R35 +8.33; 128-hop Δ vs immediate ablation +0.00.

## Следующий шаг

R43: stop residual-gating variants and test direct set-level listwise reranking over the retained beam, because branch-wise scalar corrections have now failed across static caps, look-ahead, immediate pairwise and multi-step value targets.
