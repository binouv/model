# FlyGraph R56 — protected-address diversity allocation

**Статус: REFUTED**

## Точная гипотеза

R54 уже показывает правильный ECC address в top-k достаточно часто, но несколько высокоскоринговых trajectories с одним и тем же текущим protected address могут crowd-out альтернативные адресные гипотезы. Поэтому R56 группирует кандидатов по **hard-decoded protected address следующего pointer** и перед global backfill допускает не более `cap` trajectories из одной address-группы.

Protected address остаётся дискретным и отделённым от reasoning workspace; soft posterior mixing не используется.

## Одна подтверждающая абляция

Тот же frozen R35 scorer, тот же R52 ECC, тот же R54 uncertainty trigger и тот же adaptive ECC candidate support, но обычный **global top-32 pruning** без diversity allocation.

## Calibration

- calibration seeds: `5621, 5622`
- held-out seeds: `5651, 5652, 5653`
- cap grid: `1, 2, 4, 8`
- calibration conditions: `8` и `64` hops, по одному полностью завершённому episode на seed/condition
- выбран **cap = 8**
- никаких held-out результатов при выборе cap не использовалось

На calibration cap=8 сохранил 100% 8-hop answer и лучший 64-hop path survival/final-path среди проверенных cap. Более агрессивные cap=1/2/4 резко повышали diversity, но ухудшали survival истинной trajectory.

## Completed held-out cognition metrics

| Hops | N | **R56 address-diverse** | R54 global-top32 ablation | Δ answer | R56 path survival | Ablation path survival |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 18 | **100.00%** | 100.00% | +0.00 pp | 100.00% | 100.00% |
| 32 | 12 | **100.00%** | 100.00% | +0.00 pp | 100.00% | 100.00% |
| 64 | 12 | **83.33%** | 83.33% | +0.00 pp | 96.74% | 96.74% |
| 128 | 6 | **33.33%** | 33.33% | +0.00 pp | 43.36% | 43.36% |

Semantic mismatches: **0**.

## Что изменилось внутри beam

R56 действительно сработал как diversity mechanism:

- 64 hops: mean unique address groups в beam `12.10 → 13.13`;
- 64 hops: largest address-group fraction `31.78% → 21.39%`;
- 128 hops: unique groups `12.92 → 13.46`;
- 128 hops: largest group fraction `26.52% → 20.62%`;
- diversity pruning реально менял membership примерно в **29.17%** шагов на 64 hops и **18.23%** на 128 hops.

Но ни answer, ни exact true-path survival от этого не улучшились. Candidate compute практически не изменился, то есть результат нельзя объяснить меньшим/большим compute budget.

## Вердикт

**REFUTED.** Address-level crowding существует, но **не является текущим причинным bottleneck cognition accuracy**. Простое увеличение числа разных текущих адресов в beam не спасает exact reasoning trajectory.

Особенно показательно, что на 64 hops R56 существенно уменьшает доминирование одной address-группы, однако answer остаётся **83.33%** и полностью совпадает с абляцией. На 128 hops также нет выигрыша: **33.33% vs 33.33%**.

Это означает, что несколько trajectories, пришедших в один и тот же address, **не эквивалентны по reasoning history/state**. Address-only grouping слишком грубое: оно уменьшает redundancy, но не выделяет causal distinction, необходимую для дальнейшего reasoning.

## Следующий шаг

**R57: state-conditional recombination/diversity по `(decoded protected address, reasoning state)`**. Идея: ограничивать дубликаты только когда trajectories сошлись в один и тот же actionable latent state, но не заставлять разные reasoning states конкурировать только потому, что у них совпал address.

Единственная абляция R57: текущий R54 global top-32 на идентичном adaptive candidate support. Parameter scaling 100M/200M/300M всё ещё откладывается: bottleneck остаётся hypothesis selection / memory uncertainty.