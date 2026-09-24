# FlyGraph R60 — selective exact fallback

**Статус: REFUTED / SPURIOUS ANSWER GAIN**

## Точная гипотеза
A fixed compute-capped oracle-free trigger based only on the R59 student rerank boundary margin can spend exact R58 H=1 lookahead on the most ambiguous ~25% of steps, improving long-horizon robustness without losing short-horizon accuracy.

## Calibration
- Separate seeds: `6041/6042`.
- 25% quantile threshold: **0.0214364938712599** over 890 student decision steps.

## Held-out
| Hops | N | R60 answer | R59 answer | R60 final true path | R59 final true path | Trigger | Compute ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 12 | 100.00% | 100.00% | 100.00% | 100.00% | 10.7% | 1.190× |
| 64 | 12 | 75.00% | 75.00% | 75.00% | 75.00% | 23.4% | 1.373× |
| 128 | 6 | 50.00% | 50.00% | 50.00% | 50.00% | 24.0% | 1.370× |
| 256 ECC | 6 | 16.67% | 0.00% | **0.00%** | **0.00%** | 25.2% | 1.369× |

## Memory error decomposition
256 generation losses/episode: R60 **0.333**, R59 **0.333**.
256 pruning losses/episode: R60 **0.667**, R59 **0.667**.
256 path survival: identical, **0.05143**.

## Вердикт
**REFUTED / SPURIOUS ANSWER GAIN.** The +16.67 pp 256-hop answer gain is not trajectory recovery: final true-path retention, path survival, generation losses and pruning losses are unchanged. A wrong trajectory converged to the same 16-state target, so the answer-only collision is explicitly not counted as a cognition improvement.

Semantic mismatches: **0**. Only completed chunks included.

## Следующий шаг
R61: before-pruning discrete address-support expansion (top4→top8) only for highly ambiguous parent pointers; ablation is the frozen top4 policy.
