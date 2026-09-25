# FlyGraph R61 — learned pre-pruning top4→top8 address-support gate

**Статус:** REFUTED / MIXED

## Точная гипотеза

A trained oracle-free gate can identify parent pointers where the correct discrete ECC address falls outside top4 but inside top8, expanding address support before pruning only on those ambiguous parents. This should reduce long-horizon generation losses while preserving the frozen R59 branch-value policy and short-horizon cognition.

## Обучение

- Gate train rows: **1368 × 24**, positives **38 (2.78%)**.
- Calibration threshold: **0.858955**; true-pointer trigger rate **5.59%**, recall **65.00%**.
- Frozen R59 student was reconstructed deterministically; its original held results were reproduced exactly (8=100%, 64=100%, 128=83.33%).
- Единственная абляция: frozen top-4 ECC support; all other candidate generation, R59 value ranking, seeds and budgets are identical.

## Completed held-out cognition

| Hops | N | **R61 answer** | Top4 ablation | R61 final path | Top4 final path | R61 path survival | Top4 survival | Gen loss R61 / top4 | Compute ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 12 | **100.00%** | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.000 / 0.000 | 1.035× |
| 64 | 12 | **83.33%** | 83.33% | 91.67% | 91.67% | 93.75% | 93.75% | 0.000 / 0.000 | 1.076× |
| 128 | 6 | **50.00%** | 50.00% | 50.00% | 50.00% | 61.85% | 61.85% | 0.000 / 0.000 | 1.088× |
| 256 | 6 | **33.33%** | 33.33% | 16.67% | 16.67% | 23.76% | 23.76% | 0.167 / 0.167 | 1.120× |

## Вердикт

**REFUTED / MIXED**
- 8-hop answer Δ: **+0.00 п.п.**
- 128 final-path Δ: **+0.00 п.п.**
- 256 final-path Δ: **+0.00 п.п.**
- 256 path-survival Δ: **+0.00 п.п.**
- 256 generation losses reduced: **+0.000/episode**
- 256 compute ratio: **1.120×**

## Следующий шаг

R62: the remaining support failure is not solved by a pointwise top4→top8 gate; train a parent-level expected-value-of-expansion policy that predicts downstream trajectory survival benefit of spending extra address candidates, rather than decoder rank alone. Keep R59 frozen and compute-matched.
