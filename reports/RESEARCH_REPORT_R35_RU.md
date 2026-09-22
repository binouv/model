# FlyGraph R35 — consistency verifier / reranker for multi-hypothesis memory

**Дата:** 2026-09-22  
**Статус:** completed  
**Приоритет:** cognition — память, reasoning, выбор траектории. Audio/video не затрагивались.

## Точная гипотеза

В noisy mutable memory при `p=0.30` beam должен ранжироваться не только суммой локальных retrieval logits. Небольшой verifier, использующий только доступные на inference признаки согласованности траектории, должен улучшить 32/64-hop reasoning и при этом не ухудшить 8-hop accuracy более чем на 1–2 п.п. относительно greedy.

Verifier использует: similarity текущего subject к pointer, relation-program match, recency, local log-probability, retrieval margin/entropy, признак latest-write для `(subject, relation)`, repeated-key consistency, object→next-subject compatibility для следующей relation, compatibility margin и накопленную trajectory confidence. Oracle `idx` применяется только как training/eval label и не входит в inference features.

## Протокол

| Часть | Значение |
|---|---|
| Noise | bit-flip `p=0.30` |
| Train seeds | 3501, 3502, 3503 |
| Calibration seed | 3599 |
| Held-out seeds | 3601, 3602, 3603 |
| Train episodes | 24 / seed / condition |
| Held-out episodes | 16 / seed / condition = 48 на condition |
| Beam | width 8, expand 8 |
| Learned verifier | balanced logistic reranker, 59,904 candidate rows |
| Calibration | `local_alpha ∈ {0, 0.25, 0.5}`, выбран 0.25 |
| Baselines | greedy, fixed beam8 local-likelihood, R34 adaptive beam |
| Confirming ablation | verifier, обученный только на local retrieval features без explicit consistency features |

Все основные held-out результаты ниже завершены; незавершённые прогоны в метрики не включались.

## Основной held-out результат

Answer accuracy, mean по 3 held seeds:

| Hops | Greedy | Beam8 local | R34 adaptive | **Beam8 + verifier** |
|---:|---:|---:|---:|---:|
| 8 | 97.92% | 91.67% | 93.75% | **100.00%** |
| 32 | 47.92% | 60.42% | 58.33% | **89.58%** |
| 64 | 14.58% | 35.42% | 35.42% | **66.67%** |
| 128 | 4.17% | **16.67%** | 0.00% | 14.58% |

На 8 hops short-task regression отсутствует: verifier даёт `48/48`, greedy `47/48`. На 32 hops verifier улучшает greedy на **+41.67 п.п.**, fixed beam8 на **+29.17 п.п.**, R34 adaptive на **+31.25 п.п.**. На 64 hops прирост составляет **+52.08 / +31.25 / +31.25 п.п.** соответственно.

## Memory / trajectory metrics

| Hops | Method | Mean true-path survival over steps | True path in final beam | Selected trajectory exactly true |
|---:|---|---:|---:|---:|
| 8 | Greedy | 99.22% | 97.92% | 97.92% |
| 8 | **Verifier** | **100.00%** | **100.00%** | **100.00%** |
| 32 | Greedy | 71.09% | 39.58% | 39.58% |
| 32 | Beam8 local | 87.11% | 77.08% | — |
| 32 | **Verifier** | **96.42%** | **93.75%** | **87.50%** |
| 64 | Greedy | 54.46% | 10.42% | 10.42% |
| 64 | Beam8 local | 52.25% | 33.33% | — |
| 64 | **Verifier** | **78.48%** | **68.75%** | **64.58%** |
| 128 | Greedy | 23.78% | 0.00% | 0.00% |
| 128 | Beam8 local | 10.92% | 0.00% | — |
| 128 | **Verifier** | **26.82%** | **14.58%** | **10.42%** |

R35 не просто меняет final-state voting: verifier намного чаще сохраняет правильную memory trajectory. Это особенно выражено на 64 hops, где final true-path retention вырос с 33.33% у local beam до 68.75%.

## Подтверждающая абляция

Удалены explicit consistency features: `latest-write`, repeated-key, next-relation object→subject compatibility и future-compat accumulation. Оставлен обучаемый reranker на локальных retrieval statistics.

| Hops | Local-only learned verifier | Full consistency verifier | Δ full |
|---:|---:|---:|---:|
| 8 | 97.92% | **100.00%** | +2.08 п.п. |
| 32 | 70.83% | **89.58%** | +18.75 п.п. |
| 64 | 22.92% | **66.67%** | +43.75 п.п. |

Это подтверждает, что основной gain на длинных цепочках приходит не просто от «ещё одного обученного scorer», а от trajectory/memory consistency information.

## Extension: 128 / 256 hops

128-hop уже входил в основной held-out. Здесь verifier сохраняет true path существенно лучше baseline (`14.58%` final true path против `0%` у local beam), но answer accuracy `14.58%` не превосходит local beam `16.67%`. Следовательно, ranking ещё недостаточно стабилен на 128 hops.

Для 256 hops выполнен отдельный завершённый small-N extension: 3 held seeds × 8 episodes = 24 episodes, condition `(hops=256, entities=576, edges=1024, update_prob=.50, p=.30)`.

| Method | 256-hop answer | Mean path survival | Final true path in beam |
|---|---:|---:|---:|
| Greedy | 4.17% | 11.39% | 0.00% |
| Beam8 local | 8.33% | 3.86% | 0.00% |
| R34 adaptive | 8.33% | 3.45% | 0.00% |
| Beam8 + verifier | **16.67%** | 9.23% | 0.00% |

256-hop answer gain **не считается успешным reasoning result**: ни один метод не сохранил exact true path до конца, алфавит final state содержит только 16 состояний (`6.25%` chance), а N=24 мал. Это extension для диагностики, а не доказательство long-horizon solution.

## Вердикт

**Основной критерий R35 подтверждён.** Consistency verifier резко улучшил 32/64-hop noisy-memory reasoning и не вызвал short-task regression. Текущий bottleneck действительно был trajectory ranking, и explicit memory-consistency features его существенно уменьшают.

**Но достаточность verifier для 128/256 hops не подтверждена.** На 128 hops он лучше сохраняет корректную траекторию, однако final answer ещё не стабильно превосходит local beam; на 256 hops exact path полностью теряется.

Следовательно, возвращаться к 100M/200M/300M scaling пока рано. Следующий цикл должен использовать verifier uncertainty как управляющий сигнал для adaptive beam-width / compute allocation и, вероятно, более сильного prefix verifier с lookahead, при этом отдельно удерживая 8-hop accuracy. Только после устойчивого улучшения 128-hop и появления ненулевого exact-path survival на 256 hops имеет смысл снова масштабировать sparse workspace.

## Артефакты

`r35_consistency_verifier.py` — полностью самодостаточный benchmark/train/eval script.  
`r35_metrics.json` — полный protocol, coefficients, calibration и held metrics.  
`r35_metrics.csv` — компактная таблица результатов.  
`r35_verifier_weights.npz` — веса reranker.  
`r35_config.json` — фиксированный protocol.  
`test_r35_semantics.py` — semantic/feature sanity tests; PASS.  
`r35_256_extension.json` — отдельный completed 256-hop extension.