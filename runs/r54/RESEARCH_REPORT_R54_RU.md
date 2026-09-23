# FlyGraph R54 — adaptive uncertainty-triggered ECC branching

**Статус:** REFUTED / MIXED

## Точная гипотеза

Trigger discrete ECC multi-hypothesis expansion only when decoder uncertainty is high, while preserving the exact R52/R35 raw candidate support on confident steps, will recover the true-address benefit of R53 without its branch-budget collapse and thereby improve 64/128-hop reasoning at lower compute than always-on top-4 branching.

## Протокол

- Главный приоритет: cognition / reasoning / memory. Audio/video/body не менялись.
- Frozen R52 protected address: binary linear `[96,9]` ECC, 512 codewords, minimum distance **41**.
- Frozen R35 verifier: beam **32**, edge expand **8**.
- На каждом hypothesis-step R54 **всегда сохраняет обычную R52/R35 raw candidate support**.
- Дополнительные top-4 ECC branches открываются только если `top1-top2` decoder log-posterior margin <= **1.694594**.
- Порог выбран только на calibration seeds `5421/5422`: F1=0.597 для обнаружения hard top-1 decoder errors, recall=0.812, precision=0.472, trigger rate=28.56%. Held-out accuracy для выбора порога не использовалась.
- Held-out seeds: `5451/5452/5453`; episode counts: 18 / 12 / 12 / 6 для 8/32/64/128 hops.
- Единственная абляция: **always-on R53 top-4 branching** с тем же frozen codebook/scorer и `beta=.10`.

## Completed held-out memory/reasoning

| Hops | N | R52 ECC + R35 | **R54 adaptive** | Always-on top-4 ablation | R54 path survival | R54 true addr top-4 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 18 | 100.00% | **100.00%** | 100.00% | 100.00% | 97.92% |
| 32 | 12 | 100.00% | **100.00%** | 83.33% | 100.00% | 95.05% |
| 64 | 12 | 91.67% | **91.67%** | 75.00% | 92.84% | 94.01% |
| 128 | 6 | 16.67% | **16.67%** | 16.67% | 39.97% | 95.05% |

## Adaptive compute

| Hops | R52 evals | R54 evals | Always top-4 evals | R54 reduction vs always-on | R54 triggered hypotheses |
|---:|---:|---:|---:|---:|---:|
| 8 | 1608 | 3367 | 6667 | 49.5% | 27.02% |
| 32 | 7752 | 15587 | 31421 | 50.4% | 24.93% |
| 64 | 15944 | 33382 | 64221 | 48.0% | 27.27% |
| 128 | 32328 | 69172 | 129883 | 46.7% | 28.34% |

## Semantic validation / completion

Semantic mismatches: **0** total; per condition: `{'id8': 0, 'ood32': 0, 'ood64': 0, 'ood128': 0}`.

The 8/32/64 batches completed fully. For 128 hops, the first shell wrapper ended after seed batches `5451` and `5452` had already been fully written. Seed `5453` was then executed separately. The table contains exactly **6 fully completed 128-hop episodes**; no partial episode or timed-out metric is included.

## Вердикт

**REFUTED / MIXED.**

- 8-hop Δ R54 vs R52: **+0.00 п.п.**
- 64-hop Δ R54 vs R52: **+0.00 п.п.**
- 128-hop Δ R54 vs R52: **+0.00 п.п.**
- 64-hop Δ R54 vs always-on top-4: **+16.67 п.п.**
- 128-hop Δ R54 vs always-on top-4: **+0.00 п.п.**

Adaptive gating **does solve the compute-allocation failure of R53 partially**: at 64 hops it preserves R52 accuracy while using 48.0% fewer candidate evaluations than always-on top-4; at 128 hops the reduction is 46.7%.

But the main accuracy hypothesis is not confirmed. R54 is effectively accuracy-neutral relative to R52 at both 64 and 128 hops. The added ECC alternatives are available only when uncertainty is high, yet ordinary global top-32 pruning still removes them before they can repair the trajectory. The new bottleneck is therefore **survival allocation among discrete branches**, not decoder coverage or trigger detection.

## Следующий шаг

R55: uncertainty-triggered protected branch reservation. Keep the R54 trigger and baseline path, but reserve a small calibrated quota of beam slots for ECC-alternative trajectories on uncertain steps so alternatives cannot be immediately pruned by the frozen R35 global score. One ablation: identical R54 trigger/candidates with ordinary global top-32 pruning and no reserved slots. Do not scale workspace parameters yet.