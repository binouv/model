# FlyGraph R55 — uncertainty-triggered protected branch reservation

**Статус:** REFUTED

## Точная гипотеза

A small reserved quota for uncertainty-triggered ECC-only branch lineages will prevent useful discrete address alternatives from being immediately eliminated by global R35 pruning, improving 64/128-hop trajectory survival and answer accuracy without hurting 8-hop cognition.

## Протокол

- Главный приоритет: cognition / reasoning / memory. Audio/video/body не менялись.
- Frozen R52 protected address: binary linear `[96,9]` ECC, 512 codewords, minimum distance **41**.
- Frozen R35 verifier; beam width **32**, edge expand **8**.
- R54 uncertainty trigger полностью frozen: `top1-top2` ECC log-posterior margin <= **1.694594**, `top_k=4`, `beta=0.10`.
- R55 маркирует только **ECC-only** edges (которых нет в raw R35 top-8 support) как protected branch lineage.
- Protection живёт **2 pruning decisions**. На каждом шаге до `2` из 32 beam slots резервируются для лучших по обычному score живых protected descendants; остальные slots заполняются обычным global ranking.
- Quota grid `2/4/8` проверен только на calibration seeds `5521/5522`; все варианты дали одинаковый calibration objective, поэтому по tie-break выбран минимальный quota **2**.
- Held-out seeds `5551/5552/5553`; N = 18 / 12 / 12 / 6 для 8/32/64/128 hops.
- Единственная абляция: тот же R54 raw+ECC candidate union и uncertainty trigger, но **без reservation**, обычный global top-32 prune.
- R52 ECC+R35 показан только как frozen reference на тех же эпизодах.

## Completed held-out memory/reasoning

| Hops | N | R52 reference | **R55 reserved** | R54 no-reservation ablation | R55 path survival | True prefix protected |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 18 | 100.00% | **100.00%** | 100.00% | 100.00% | 0.00% |
| 32 | 12 | 100.00% | **100.00%** | 100.00% | 100.00% | 0.00% |
| 64 | 12 | 66.67% | **75.00%** | 75.00% | 80.99% | 0.52% |
| 128 | 6 | 66.67% | **66.67%** | 66.67% | 70.44% | 0.00% |

## Что произошло с reservation

На 64 hops protected branches занимали в среднем **6.60%** beam, но exact true prefix был в protected stratum всего **0.52%** шагов. На 128 hops protected occupancy была **6.47%**, а exact true prefix — **0.00%** шагов.

То есть R55 действительно держал отдельные ECC-only alternatives, однако это почти никогда не были те trajectories, которые требовалось спасать. Поэтому main и no-reservation ablation дали идентичную long-horizon accuracy и survival.

## Вердикт

**REFUTED.**

- 8-hop Δ R55 vs R54: **+0.00 п.п.**
- 64-hop Δ R55 vs R54: **+0.00 п.п.**
- 128-hop Δ R55 vs R54: **+0.00 п.п.**
- 64-hop Δ R55 vs R52 reference: **+8.33 п.п.**
- 128-hop Δ R55 vs R52 reference: **+0.00 п.п.**

Главная гипотеза отклонена: проблема **не в немедленном global pruning именно ECC-only alternatives**. R53/R54 показали, что decoder обычно содержит правильный address в top-k, а R55 теперь показывает, что правильный exact prefix почти всегда находится в обычной raw candidate support, а не в специальной ECC-only ветке. Следующий bottleneck — **crowding/ranking между множеством дискретных trajectories внутри общей protected-address support**.

## Integrity / completion

Semantic mismatches: **0** total; `{'id8': 0, 'ood32': 0, 'ood64': 0, 'ood128': 0}`.

Перед финальными runs была обнаружена implementation-инварианта в exploratory версии: retained reserved beam не сортировался обратно по global score, из-за чего protected element ошибочно становился output. Эти exploratory результаты удалены. После исправления calibration и все held-out conditions были выполнены заново. Один параллельный wrapper позже завершился по wall-time; итог агрегирован только из полностью записанных per-seed JSON файлов, перечисленных в `r55_metrics.json`.

## Следующий шаг

**R56: protected-address diversity allocation.** Вместо резервирования только ECC-only lineage группировать живые trajectories по decoded current protected-address identity и ограничивать/стратифицировать число почти дублирующих branches на один address, чтобы несколько высоко scored ложных trajectories не съедали весь beam. Единственная абляция — обычный R54/R35 global top-32 на тех же candidates. Scaling 100M/200M/300M пока не запускать.