# FlyGraph R58 — shallow lookahead value reranking

**Статус: CONFIRMED**

## Точная гипотеза

A shallow oracle-free future-consistency/value estimate at the pruning boundary can rescue the correct protected trajectory when its immediate R35 score is temporarily lower than a false branch. This should improve 64/128-hop reasoning without short-horizon regression.

## Протокол

- Главный приоритет: cognition / reasoning / memory. Audio/video не менялись.
- Frozen R52 protected ECC address + frozen R35 verifier/scorer; R54 candidate support unchanged.
- Beam width 32, expand 8, ECC top-k 4, beta 0.10, frozen uncertainty threshold 1.694594383239746.
- Перед pruning только immediate top-48 candidates получают oracle-free shallow future value.
- Horizon grid 1/2/4 проверен только на calibration seeds 5821/5822. Все три дали одинаковый objective; tie-break выбрал H=1.
- Единственная абляция: тот же R54 candidate support + обычный frozen R35 global top-32 без lookahead.
- Held seeds 5851/5852/5853 не участвовали в выборе horizon. Незавершённые runs не включались.

## Completed held-out memory/reasoning

| Hops | N | **R58 H=1 answer** | R54 ablation | Δ | R58 path survival | Ablation survival | R58 prune loss/ep | Ablation prune loss/ep |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 18 | **100.00%** | 100.00% | +0.00 п.п. | 100.00% | 100.00% | 0.000 | 0.000 |
| 32 | 12 | **100.00%** | 100.00% | +0.00 п.п. | 100.00% | 100.00% | 0.000 | 0.000 |
| 64 | 12 | **91.67%** | 83.33% | +8.33 п.п. | 94.79% | 87.76% | 0.000 | 0.083 |
| 128 | 6 | **83.33%** | 66.67% | +16.67 п.п. | 84.24% | 71.22% | 0.000 | 0.167 |

## Causal diagnostics

- 64 hops: final true-path `91.67%` vs `83.33%`; pruning losses `0.000` vs `0.083` per episode.
- 128 hops: final true-path `83.33%` vs `66.67%`; pruning losses `0.000` vs `0.167` per episode.
- 128 hops decoder true-address top-4 coverage is identical: `94.66%`. Generation losses are also identical at `0.167`/episode. Therefore the gain is not from improved address decoding/support.
- Lookahead changed beam membership on `94.79%` of 128-hop steps, produced `0.333` rescue events/episode and **0 observed harm events**.
- Cost: 128-hop total edge evaluations `176144` vs `71809` = **2.45×**. The gain is real but currently compute-expensive.

## Semantic / run integrity

Semantic mismatches: **0**. 128-hop aggregate contains only fully completed seed chunks `5851/5852/5853`. No timed-out or partial grid is counted. One later full-rerun wrapper hit wall-time during ood64 and is explicitly excluded; final aggregation uses only individually completed condition/seed files.

## Вердикт

**CONFIRMED.** Predefined criterion is met: 8-hop regression +0.00 п.п.; 64-hop gain +8.33 п.п.; 128-hop gain +16.67 п.п.

R58 improves the exact failure mode isolated by R57: correct extensions are usually generated, but immediate trajectory score sometimes prunes the correct path. One-step lookahead eliminated observed pruning losses on the fresh 64/128 held sets. Remaining failures are generation/support misses, not observed prune-boundary losses.

The 128-hop sample is small (N=6), so the magnitude of the gain is not yet a stable population estimate. The causal direction is nevertheless supported by the matched pruning-loss diagnostic on the held episodes.

## Следующий шаг

R59: compute-efficient learned/heuristic value cache distillation. Preserve the successful R58 H=1 ranking signal, but predict its one-step future value from current candidate/ancestry features to approach R54 compute while retaining R58 pruning gains. One ablation: exact R58 H=1 lookahead. Do not scale 100M/200M/300M yet.

