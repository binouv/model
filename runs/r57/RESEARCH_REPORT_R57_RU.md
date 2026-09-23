# FlyGraph R57 — state-conditional diversity over protected address × reasoning state

**Статус:** REFUTED

## Точная гипотеза

Beam crowding может возникать не просто по protected address, а среди trajectories, которые пришли в один и тот же **actionable latent state**. Поэтому pruning должен ограничивать occupancy по совместному ключу:

`(hard-decoded protected address, next reasoning state)`

Если это причинный bottleneck, такой pruning должен повысить long-horizon true-path survival / answer по сравнению с обычным global top-32, не ухудшая короткий горизонт.

## Протокол

- Главный приоритет: memory / reasoning / intelligence.
- R52 ECC frozen, R35 verifier/scorer frozen.
- Candidate generation полностью совпадает с R54 adaptive ECC branching.
- Beam width = 32, expand = 8, ECC top-k = 4, beta = 0.10.
- Trigger threshold = `1.694594383239746`, frozen из R54.
- Calibration seeds: `5721, 5722`.
- Held-out seeds: `5751, 5752, 5753`.
- Calibration cap grid: `1,2,4,8`; все варианты дали одинаковый objective, поэтому по заранее заданному tie-break выбран минимальный `cap=1`.
- Единственная абляция: тот же candidate set + frozen R35 score + обычный global top-32.
- Oracle information при inference не используется.
- Незавершённые runs в итоговые metrics не включались.

## Completed held-out memory/reasoning

| Hops | N | **R57 joint-state diverse** | R54 global top-32 | Δ answer | R57 path survival | Ablation survival |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 18 | **94.44%** | 94.44% | +0.00 п.п. | 100.00% | 100.00% |
| 32 | 12 | **100.00%** | 100.00% | +0.00 п.п. | 100.00% | 100.00% |
| 64 | 12 | **100.00%** | 100.00% | +0.00 п.п. | 100.00% | 100.00% |
| 128 | 6 | **66.67%** | 66.67% | +0.00 п.п. | 73.83% | 73.83% |

## Что реально изменилось в beam

Механизм diversity был активным, а не тривиальным:

- 64 hops: unique joint-state groups `26.42 → 31.69`.
- 64 hops: крупнейшая joint-state group `8.26% → 3.24%`.
- 64 hops: membership beam менялся в `54.56%` pruning steps.
- 128 hops: unique joint-state groups `28.75 → 31.86`.
- 128 hops: крупнейшая group `6.94% → 3.17%`.
- 128 hops: membership менялся в `41.54%` pruning steps.

Несмотря на это, answer и true-path survival остались полностью идентичными абляции.

## Дополнительная causal diagnostic

На 128 hops:

- decoder true-address top-4 coverage = **95.18%**;
- среднее число true-extension generation losses = **0.000 на episode**;
- среднее число true-path pruning losses = **0.333 на episode**.

То есть когда exact true prefix ещё жив, его правильное продолжение **генерировалось** в candidate set во всех наблюдавшихся случаях. Потери происходили на pruning/ranking, а не из-за отсутствия правильного кандидата. Joint `(address,state)` diversity эти pruning losses не исправила.

## Вердикт

**REFUTED.**

R57 почти полностью устранил duplicate occupancy по `(address, reasoning state)` и существенно менял beam membership, но не дал ни одного пункта прироста answer или path survival.

Это опровергает гипотезу, что текущая ошибка объясняется crowding даже на уровне совместного protected address + reasoning state.

На текущих 128-hop runs наблюдается более точная картина: правильная trajectory присутствует среди кандидатов до pruning, но иногда проигрывает **другому, отличному actionable state** по текущему trajectory score. Следовательно, bottleneck снова сужается к **value/ranking на prune boundary**, а не к decoder coverage или duplicate occupancy.

## Следующий шаг

**R58: shallow lookahead value reranking.**

Сохраняем protected discrete trajectories, R54 candidate support и frozen R35. Перед top-32 pruning каждому кандидату добавляется ограниченный future-consistency/value estimate из короткого rollout horizon `1/2/4`, выбранного только на calibration seeds. Это проверит, может ли небольшой прогноз будущей согласованности спасти правильный trajectory, когда его immediate score ниже ложной ветки.

Единственная абляция: идентичный candidate set с обычным R35 global top-32.

До стабилизации hypothesis selection **100M/200M/300M scaling не запускается**.