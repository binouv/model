# FlyGraph R27–R30 — benchmark integrity, discrete memory routing и scaling ассоциативной памяти

**Дата:** 2026-09-22  
**Runtime:** Python 3.13 / PyTorch CPU; CUDA недоступна в этом runtime.  
**Статус:** все метрики ниже получены фактическими запусками. Неоконченный full-grid primitive experiment не используется как evidence.

## 1. Почему пришлось остановить прежнее масштабирование

R26 предполагал, что learned transition operator ошибается при глубокой композиции. R27 показал, что одношаговый оператор из фактического `r26_hard_state.pt` имеет **100% top-1 single-step accuracy**. Проверка генераторов выявила более фундаментальную проблему: повторяющиеся `(entity, relation)` ключи могли быть перезаписаны позже, но target/trace продолжали описывать старую цепочку.

### Старый R25/R26 computation benchmark: доля полностью согласованных примеров

| hops | fully consistent |
|---:|---:|
| 8 | 67.71% |
| 16 | 39.11% |
| 32 | 9.81% |
| 64 | 0.217% |
| 128 | 0.00% |

R23/R24 pointer benchmark имел тот же класс дефекта: fully-consistent = 63.41% / 34.07% / 4.82% / 0.043% для 8/16/32/64 hops. Поэтому старые **абсолютные long-hop результаты R23–R26 invalidated**. Исходные файлы сохранены для provenance, но не должны использоваться как текущая оценка архитектуры.

## 2. R27 — исправленный benchmark v2 и повторная проверка R26

Исправление: path source keys уникальны; stale/distractor writes идут до authoritative chain writes; `edge_idx`, trace, next entity и target вычисляются из одной финальной памяти.

Фактический R26 transition primitive:
- single-step accuracy: **100%**;
- NLL: **0.0262**;
- hard-state operator-only composition: **100% до 256 hops**;
- soft-state composition: 100% @32, 99.95% @64, 88.38% @128, 8.84% @256.

На corrected benchmark actual R26 checkpoint с **oracle read или hard top-1 read = 100% answer accuracy на 8/16/32/64/128 hops**. Следовательно, прежний вывод «оператор сам по себе не умеет глубокую композицию» был неверен; реальный remaining failure — probability/read dilution при soft state/read.

### Повтор R23 identity-vs-monolithic на corrected benchmark

Matched train: 140 updates × batch16, seeds 27011/27012/27013.

| condition | monolithic | protected identity | identity + hard read |
|---|---:|---:|---:|
| id8 | 12.85±4.21% | **100.00±0.00%** | **100.00%** |
| ood16 | 11.81±1.20% | **99.65±0.60%** | **100.00%** |
| ood32 | 13.19±1.20% | **98.26±1.59%** | **100.00%** |
| ood64 | 12.15±3.01% | **100.00±0.00%** | **100.00%** |
| distractor8 | 10.76±2.62% | **97.57±2.41%** | **100.00%** |

Качественный вывод R23 **реплицирован**: protected identity/address stream существенно лучше monolithic recurrent state. Теперь это подтверждено на согласованном benchmark.

## 3. R28 — что именно ломает длинную память

Использован corrected computation benchmark, actual R26 transition operator и 3 eval seeds. Hard workspace state фиксирует reasoning primitive; меняется только read policy.

| policy | 16 hops | 64 hops | 128 hops |
|---|---:|---:|---:|
| soft both | 98.44% | 97.40% | 91.15% |
| hard address only | 98.44% | 97.92% | 93.75% |
| hard value only | 100.00% | 100.00% | 96.88% |
| sharpen ×2 | 100.00% | 99.48% | 97.40% |
| sharpen ×4 | 100.00% | 100.00% | 99.48% |
| top-k=2 | 100.00% | 98.44% | 96.88% |
| hard both | 100.00% | 100.00% | 100.00% |

При latent address noise:

| σ | soft @64/128 | sharpen×4 @64/128 | hard both @64/128 |
|---:|---:|---:|---:|
| 0.05 | 97.22%/85.42% | 100.00%/98.61% | 100.00%/100.00% |
| 0.10 | 80.56%/27.08% | 100.00%/95.14% | 100.00%/100.00% |
| 0.20 | 5.56%/1.39% | 41.67%/8.33% | 54.17%/10.42% |

**Вывод R28:** до умеренного address-noise bottleneck — не отсутствие вычислительной мощности, а soft retrieval/state diffusion. При σ=0.20 уже ломается сама identity/address representation, и hard top-1 перестаёт спасать.

## 4. R29 — настоящее 100M→200M→300M scaling на capacity-sensitive memory task

Чтобы не «доказывать scaling» на уже решённом R27 task, введён отдельный synthetic episodic associative benchmark.

- random normalized key (16D) → random label (128 classes);
- fixed SimHash router, 24 bits;
- один active expert `16×128` на query = **2,048 active parameters** для всех размеров;
- physical capacity: 99,999,744 / 199,999,488 / 299,999,232 параметров;
- phase A: 524,288 unique associations, один проход;
- phase B: ещё 524,288 новых associations, один проход;
- одинаковые data seeds и presentation order для 100/200/300M;
- independent seeds: **2901/2902/2903**;
- no optimizer state; sparse row-SGD обновляет только routed expert rows.

| physical model | A after A | A retained after B | B after B | A forgetting |
|---|---:|---:|---:|---:|
| ~100M | {ms(100,'A_after_A_exact')} | {ms(100,'A_after_B_retained')} | {ms(100,'B_after_B_exact')} | {r29['scales']['100']['metrics']['A_forgetting_pp']['mean']:.2f}±{r29['scales']['100']['metrics']['A_forgetting_pp']['sd']:.2f} pp |
| ~200M | {ms(200,'A_after_A_exact')} | {ms(200,'A_after_B_retained')} | {ms(200,'B_after_B_exact')} | {r29['scales']['200']['metrics']['A_forgetting_pp']['mean']:.2f}±{r29['scales']['200']['metrics']['A_forgetting_pp']['sd']:.2f} pp |
| ~300M | {ms(300,'A_after_A_exact')} | {ms(300,'A_after_B_retained')} | {ms(300,'B_after_B_exact')} | {r29['scales']['300']['metrics']['A_forgetting_pp']['mean']:.2f}±{r29['scales']['300']['metrics']['A_forgetting_pp']['sd']:.2f} pp |

300M против 100M:
- +{100*(r29['scales']['300']['metrics']['A_after_A_exact']['mean']-r29['scales']['100']['metrics']['A_after_A_exact']['mean']):.2f} pp initial A recall;
- +{100*(r29['scales']['300']['metrics']['A_after_B_retained']['mean']-r29['scales']['100']['metrics']['A_after_B_retained']['mean']):.2f} pp retained A;
- +{100*(r29['scales']['300']['metrics']['B_after_B_exact']['mean']-r29['scales']['100']['metrics']['B_after_B_exact']['mean']):.2f} pp B recall;
- forgetting меньше на {r29['scales']['100']['metrics']['A_forgetting_pp']['mean']-r29['scales']['300']['metrics']['A_forgetting_pp']['mean']:.2f} pp.

Это **подтверждённый capacity scaling**, потому что active compute и sample budget совпадают; увеличивается только physical sparse memory и падает число конкурирующих фактов на bucket. Но рост не линейный и уже показывает diminishing returns.

### Новый bottleneck: noisy content routing

После phase A при σ=0.05:
- content-routed recall: **{ms(100,'A_after_A_noise05_content')} → {ms(300,'A_after_A_noise05_content')}**;
- если принудительно сохранить правильный route (oracle route): **{ms(100,'A_after_A_noise05_oracle')} → {ms(300,'A_after_A_noise05_oracle')}**.

То есть дополнительная capacity почти не лечит noisy routing. Storage representation остаётся устойчивой, если выбран правильный expert.

## 5. R30 — routing audit и отрицательный multi-probe result

Для исходного 24-bit SimHash router при σ=0.05:
- mean Hamming distance = **{r30['noise']['0.05']['hamming_mean']:.2f} bits**;
- exact bucket сохраняется только **{100*r30['noise']['0.05']['route_same_100m']:.2f}%** запросов;
- все реально flipped bits лежат среди 6 lowest-margin bits в **{100*r30['noise']['0.05']['all_flips_in_6_lowest_margin']:.2f}%**, среди 8 — **{100*r30['noise']['0.05']['all_flips_in_8_lowest_margin']:.2f}%**.

Проверен margin-aware multi-probe: enumerate 2^k соседних hash-кандидатов и выбрать expert по similarity к mean centroid bucket. На 100M, seed2901, σ=0.05 after A:
- baseline content route: **{100*mp['after_A']['0.05']['baseline_content']:.2f}%**;
- oracle route: **{100*mp['after_A']['0.05']['oracle_route']:.2f}%**;
- multi-probe k4: **{100*mp['after_A']['0.05']['multiprobe_k4']['answer']:.2f}%**;
- k6: **{100*mp['after_A']['0.05']['multiprobe_k6']['answer']:.2f}%**;
- k8: **{100*mp['after_A']['0.05']['multiprobe_k8']['answer']:.2f}%**.

**H-R30 refuted:** candidate generation хорошая, но mean-centroid ranking недостаточен и даже разрушает случаи, где baseline route был правильным. Нужен learned query↔memory compatibility / hierarchical router или multi-table memory, а не centroid heuristic.

## 6. Что теперь считается установленным

1. Старые R23–R26 absolute long-hop цифры больше не являются валидным evidence из-за benchmark inconsistency.
2. Protected identity stream реплицирован на corrected benchmark и остаётся сильным архитектурным решением.
3. R26 operation primitive способен на точную глубокую композицию при hard state/read; параметр reasoning-capacity не был главным bottleneck на этом тесте.
4. Soft memory/value mixing на длинном горизонте создаёт compounding error; sharpen/hard read резко улучшает результат.
5. На отдельной capacity-sensitive associative задаче 100→200→300M даёт монотонный, multi-seed подтверждённый gain и уменьшает forgetting при одинаковом active compute.
6. Текущий главный bottleneck этой memory-scaling ветки — **устойчивый content routing**, а не storage capacity.

## 7. Ограничения

- R27–R30 остаются контролируемыми synthetic benchmarks; это не оценка общей интеллектуальности.
- R29 memory bank — специализированный associative storage, не полноценный FlyGraph workspace.
- R29 три seeds достаточны для репликации тренда, но не являются scaling-law fit.
- SimHash router намеренно прост; R30 показывает его слабость при perturbations.
- R30 multi-probe выполнен на 100M / seed2901; отрицательный результат не переносится автоматически на learned multi-probe routers.
- Real audio/video в этой cognition-first ветке не использовались.

## 8. Следующий цикл R31

Приоритет: **learned robust routing без потери sparse compute**.

1. Сравнить single-table SimHash с multi-table redundant memory при matched total physical params.
2. Learned compatibility scorer должен выбирать среди небольшого candidate set; centroid-only — контроль.
3. Мерить clean capacity, σ=0.02/0.05/0.10 OOD, A→B forgetting и route recovery отдельно.
4. Только если routing становится устойчивым, встроить эту память обратно в 200M FlyGraph workspace и повторить mixed memory+reasoning benchmark.