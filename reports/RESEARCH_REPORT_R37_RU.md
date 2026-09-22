# FlyGraph R37 — off-policy hard-negative prefix verifier

**Статус:** completed  
**Приоритет:** cognition — memory/reasoning; audio/video не затрагивались.

## Точная гипотеза

A prefix/trajectory verifier trained on beam-generated off-policy hard negatives will correct remaining long-horizon ranking drift better than the same verifier trained only on teacher-forced prefixes, improving 128-hop noisy-memory accuracy while preserving 8-hop performance.

R37 проверяет именно distribution shift verifier-а: основной scorer обучен на hard negatives из ошибочных prefix states, реально посещаемых beam search. Абляция использует те же features, тот же LogisticRegression и тот же adaptive inference, но обучается только на teacher-forced правильных prefix states.

## Протокол

- Noise: bit-flip `p=0.30`.
- Train seeds: 3711–3713; calibration: 3721–3722; held-out: 3731–3733.
- Calibration: 2 episodes/seed для 8/64/128; выбран threshold `0.2` до held-out оценки.
- Held-out N: 36 (8 hops), 24 (32), 24 (64), 18 (128).
- Beam: adaptive width 8→32, expand 8.

## Held-out cognition metrics

| Hops | N | Greedy answer | R35 fixed32 | **R37 off-policy** | Teacher-only ablation | R37 final true path | R37 selected exact |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 36 | 94.44% | 100.00% | **100.00%** | 72.22% | 100.00% | 100.00% |
| 32 | 24 | 66.67% | 95.83% | **100.00%** | 12.50% | 100.00% | 95.83% |
| 64 | 24 | 20.83% | 79.17% | **87.50%** | 20.83% | 87.50% | 87.50% |
| 128 | 18 | 5.56% | 66.67% | **50.00%** | 5.56% | 38.89% | 38.89% |

## Memory / compute detail

| Hops | Method | Path survival | Final true path | Selected exact | Candidate evals/episode | Trigger rate | Trigger step |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | R35 fixed32 | 100.00% | 100.00% | 100.00% | 1608 | 0.00% | — |
| 8 | R37 off-policy | 100.00% | 100.00% | 100.00% | 461 | 2.78% | 6.0 |
| 8 | Teacher-only | 100.00% | 100.00% | 72.22% | 632 | 41.67% | 4.8 |
| 32 | R35 fixed32 | 100.00% | 100.00% | 91.67% | 7752 | 0.00% | — |
| 32 | R37 off-policy | 100.00% | 100.00% | 95.83% | 2776 | 29.17% | 17.0 |
| 32 | Teacher-only | 92.19% | 83.33% | 8.33% | 6720 | 100.00% | 6.4 |
| 64 | R35 fixed32 | 92.25% | 91.67% | 79.17% | 15944 | 0.00% | — |
| 64 | R37 off-policy | 91.60% | 87.50% | 87.50% | 9216 | 75.00% | 27.1 |
| 64 | Teacher-only | 57.75% | 41.67% | 20.83% | 15112 | 100.00% | 5.3 |
| 128 | R35 fixed32 | 63.76% | 55.56% | 55.56% | 32328 | 0.00% | — |
| 128 | R37 off-policy | 53.04% | 38.89% | 38.89% | 28968 | 100.00% | 18.5 |
| 128 | Teacher-only | 16.45% | 0.00% | 0.00% | 31357 | 100.00% | 6.1 |

## Подтверждающая абляция

Абляция резко хуже основного off-policy verifier-а: на 32 hops `12.50%` против `100%`, на 64 hops `20.83%` против `87.50%`, на 128 hops `5.56%` против `50.00%`. Следовательно, exposure к ошибочным beam-prefix states действительно нужен. Но этого недостаточно, чтобы заменить cumulative R35 score на длинном горизонте.

## Вердикт

**REFUTED / MIXED.** off-policy hard negatives clearly outperform teacher-forced prefix training, but replacing the cumulative R35 scorer with direct prefix probability regresses at 128 hops, so the predeclared criterion is not met.

- 8 hops: R37 100.00% vs greedy 94.44%: +5.56 п.п.
- 32 hops: R37 100.00% vs R35 fixed32 95.83%.
- 64 hops: R37 87.50% vs R35 fixed32 79.17%.
- 128 hops: R37 50.00% vs R35 fixed32 66.67%: -16.67 п.п. Поэтому predeclared primary criterion не выполнен.

256-hop extension **не запускался как финальный результат**, потому что основной 128-hop success criterion уже не выполнен. Незавершённый монолитный orchestration run не включён в metrics.

## Следующий шаг

R38: keep R35 cumulative trajectory score as the protected backbone and train an off-policy residual/pairwise correction that is confidence-gated rather than replacing the cumulative score. Target: retain the R37 32/64-hop gains while matching or exceeding R35 fixed32 at 128; only after that revisit 256-hop survival. 100M/200M/300M scaling remains deferred.

## Артефакты

`r37_metrics.json`, `r37_metrics.csv`, `r37_offpolicy_prefix_verifier.py`, `r37_config.json`, `r37_offpolicy_model.npz`, `r37_teacher_model.npz`, `test_r37_semantics.py`, completed chunk JSON files, frozen `r35_base.py` and `r35_verifier_weights.npz`.