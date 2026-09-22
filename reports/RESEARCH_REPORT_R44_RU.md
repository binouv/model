# FlyGraph R44 — nonlinear permutation-invariant set-context scorer

**Статус:** REFUTED / MIXED

## Точная гипотеза

A small nonlinear permutation-invariant scorer can exploit interactions among beam-relative uncertainty/support features that the linear R43 ranker misses, improving 64/128-hop noisy-memory reasoning while preserving 8-hop performance.

## Протокол

- Train [4411, 4412, 4413]; calibration [4421, 4422]; held [4441, 4442, 4443].
- Frozen protected R35 base; no R37–R42 residual.
- Main: HistGradientBoosting candidate scorer over local + permutation-invariant set-relative features.
- Единственная абляция: identical nonlinear model/data/calibration using candidate-local features only.
- Calibration selected λ main=1.00, ablation=1.00 before held.

## Held-out memory/reasoning

| Hops | N | Greedy | R35 fixed32 | **R44 nonlinear set** | Nonlinear independent | R44 final true path |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24 | 95.83% | 100.00% | **100.00%** | 100.00% | 100.00% |
| 32 | 18 | 66.67% | 94.44% | **94.44%** | 94.44% | 100.00% |
| 64 | 18 | 27.78% | 83.33% | **83.33%** | 88.89% | 88.89% |
| 128 | 12 | 0.00% | 66.67% | **75.00%** | 75.00% | 75.00% |

## Вердикт

**REFUTED / MIXED** — nonlinear set-context capacity does not satisfy the long-horizon criterion; learned scalar/set reranking is not the dominant missing mechanism.

8-hop Δ vs greedy +4.17 п.п.; 64-hop Δ vs R35 +0.00; 128-hop Δ vs R35 +8.33; 128-hop Δ vs ablation +0.00.

## Следующий шаг

R45: close learned scalar/set-reranker family and test explicit uncertainty-state propagation: maintain a compact posterior over address hypotheses across steps and merge equivalent latent states instead of repeatedly rescoring trajectories.
