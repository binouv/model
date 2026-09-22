# FlyGraph R43 — set-relative listwise beam reranking

**Статус:** REFUTED / MIXED

## Точная гипотеза

Direct set-relative reranking over the whole retained candidate beam will resolve trajectory ambiguity better than candidate-independent scoring, improving 64/128-hop noisy memory while preserving 8-hop performance.

## Протокол

- Frozen protected R35 cumulative scorer; R37–R42 residual correction disabled.
- Main: linear pairwise ranker sees candidate-local prefix features plus set-relative beam context (relative rank/gap, z-scores, edge/state support, parent rank, horizon).
- Единственная абляция: identical pairwise rows/loss/hard negatives and calibration, but only candidate-local prefix features.
- Train seeds [4311, 4312, 4313]; calibration [4321, 4322] (2 episodes/condition/seed); held [4341, 4342, 4343].
- Training: 1477 candidate sets, 17559 hard negatives, 35118 pairwise rows.
- Frozen calibration selected λ main=0.30, ablation=0.60 before held-out evaluation.

## Held-out memory/reasoning

| Hops | N | Greedy | R35 fixed32 | **R43 set-relative** | Independent ablation | R43 final true path |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24 | 79.17% | 100.00% | **100.00%** | 100.00% | 100.00% |
| 32 | 18 | 66.67% | 100.00% | **100.00%** | 100.00% | 100.00% |
| 64 | 18 | 5.56% | 66.67% | **66.67%** | 66.67% | 72.22% |
| 128 | 12 | 16.67% | 33.33% | **33.33%** | 33.33% | 41.67% |

## True-prefix rank diagnostic

| Hops | Before R43 rerank | After R43 | After independent ablation |
|---:|---:|---:|---:|
| 8 | 1.021 | 1.021 | 1.021 |
| 32 | 1.071 | 1.071 | 1.057 |
| 64 | 1.772 | 1.772 | 1.738 |
| 128 | 3.257 | 3.187 | 3.259 |

## Вердикт

**REFUTED / MIXED** — linear set-relative reranking slightly changes true-prefix rank at long horizon but does not improve held-out answer/path retention over protected R35.

8-hop Δ vs greedy +20.83 п.п.; 64-hop Δ vs R35 +0.00; 128-hop Δ vs R35 +0.00; 128-hop Δ vs ablation +0.00.

На 128 hops set-relative model сдвинул средний rank surviving true prefix примерно с 3.257 до 3.187, но это не перешло ни в answer accuracy, ни в final true-path retention. На 64 hops answer/path полностью совпали с R35. Значит линейный set-context correction недостаточен; нельзя считать bottleneck снятым.

## Следующий шаг

R44: test a small nonlinear permutation-invariant set-context scorer; use the same protected R35 generator and compare against a nonlinear candidate-independent ablation. If set context still fails to move answer/path metrics, close learned scalar/set reranking and move to explicit uncertainty-state propagation rather than more scorer capacity.
