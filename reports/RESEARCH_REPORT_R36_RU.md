# FlyGraph R36 — verifier-uncertainty adaptive beam allocation

**Статус:** completed  
**Приоритет:** cognition — memory/reasoning; audio/video не затрагивались.

## Точная гипотеза

Trajectory-verifier uncertainty can allocate beam width adaptively: width 8→32 only after a low top1-top2 trajectory verifier margin should improve 128-hop noisy-memory reasoning over R35 fixed8, preserve 8-hop accuracy within 1-2pp of greedy, and use fewer verifier-candidate evaluations than always-on width32.

R35 verifier полностью frozen. Порог trajectory-gap `0.1` выбран только на calibration seeds [3691, 3692]. После первого low-margin шага beam sticky-расширяется с width 8 до width 32.

## Протокол

- Noise: bit-flip `p=0.30`.
- Held-out seeds: 3701, 3702, 3703.
- Held N: 48 эпизодов на 8 hops; 36 на 32/64; 24 на 128; 12 на 256 extension.
- Main: verifier trajectory-gap trigger.
- Единственная подтверждающая абляция: тот же policy/threshold, но trigger по raw local retrieval margin.

## Held-out answer accuracy

| Hops | N | Greedy | R35 fixed8 | Fixed32 | **R36 adaptive** | Local-margin ablation |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 48 | 91.67% | 97.92% | 97.92% | **97.92%** | 97.92% |
| 32 | 36 | 58.33% | 91.67% | 94.44% | **91.67%** | 94.44% |
| 64 | 36 | 25.00% | 91.67% | 91.67% | **91.67%** | 91.67% |
| 128 | 24 | 12.50% | 25.00% | 62.50% | **62.50%** | 62.50% |

## Memory / trajectory metrics

| Hops | Method | Path survival | Final true path | Selected exact path | Verifier evals/episode | Trigger rate | Trigger step* |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | R35 fixed8 | 100.00% | 100.00% | 97.92% | 456 | 0.00% | — |
| 8 | Fixed32 | 100.00% | 100.00% | 97.92% | 1608 | 0.00% | — |
| 8 | R36 adaptive | 100.00% | 100.00% | 97.92% | 472 | 4.17% | 5.0 |
| 8 | Local-margin ablation | 100.00% | 100.00% | 97.92% | 1256 | 93.75% | 2.3 |
| 32 | R35 fixed8 | 99.83% | 97.22% | 91.67% | 1992 | 0.00% | — |
| 32 | Fixed32 | 100.00% | 100.00% | 91.67% | 7752 | 0.00% | — |
| 32 | R36 adaptive | 99.83% | 97.22% | 91.67% | 3672 | 41.67% | 10.0 |
| 32 | Local-margin ablation | 100.00% | 100.00% | 91.67% | 7571 | 100.00% | 1.6 |
| 64 | R35 fixed8 | 93.14% | 91.67% | 91.67% | 4040 | 0.00% | — |
| 64 | Fixed32 | 93.36% | 91.67% | 91.67% | 15944 | 0.00% | — |
| 64 | R36 adaptive | 93.19% | 91.67% | 91.67% | 10680 | 75.00% | 16.9 |
| 64 | Local-margin ablation | 93.36% | 91.67% | 91.67% | 15731 | 100.00% | 1.8 |
| 128 | R35 fixed8 | 36.49% | 25.00% | 25.00% | 8136 | 0.00% | — |
| 128 | Fixed32 | 66.24% | 54.17% | 54.17% | 32328 | 0.00% | — |
| 128 | R36 adaptive | 62.24% | 50.00% | 50.00% | 30064 | 100.00% | 12.8 |
| 128 | Local-margin ablation | 66.24% | 54.17% | 54.17% | 32224 | 100.00% | 1.0 |

*Trigger step индексируется с нуля и усредняется только по эпизодам, где расширение сработало.

## Подтверждающая абляция

На 128 hops local-margin trigger получает ту же answer accuracy 62.50% и fixed32 также 62.50%, но почти всегда расширяет beam сразу: 100.00% trigger rate и 32224 verifier evals против 30064 у verifier-gap policy. Значит trajectory-level uncertainty лучше локализует, когда дополнительный compute действительно нужен; raw retrieval uncertainty почти превращается в always-on width32.

## 256-hop extension

Завершено 12 held-out эпизодов.

| Method | Answer | Path survival | Final true path | Selected exact path | Verifier evals/episode |
|---|---:|---:|---:|---:|---:|
| Greedy | 0.00% | 11.72% | 0.00% | 0.00% | 0 |
| R35 fixed8 | 0.00% | 8.40% | 0.00% | 0.00% | 16328 |
| Fixed32 | 0.00% | 9.51% | 0.00% | 0.00% | 65096 |
| R36 adaptive | 0.00% | 8.63% | 0.00% | 0.00% | 63992 |
| Local-margin ablation | 0.00% | 9.18% | 0.00% | 0.00% | 65016 |

256 hops остаётся **не решён**: ни один из 12 held-out эпизодов не сохранил exact true path до конца, и answer accuracy у всех методов 0% в этой конкретной выборке. Дополнительная ширина не исправляет корневой ranking drift.

## Вердикт

**CONFIRMED.** verifier trajectory uncertainty is useful for adaptive compute allocation under the predeclared criterion.

- 8-hop: R36 97.92% vs greedy 91.67% — short regression отсутствует (+6.25 п.п.).
- 128-hop: R36 62.50% vs R35 fixed8 25.00%: **+37.50 п.п.**
- 128-hop compute: 30064 verifier candidate evals vs 32328 fixed32: **7.00% экономии**.
- R36 на 128 полностью совпал с fixed32 по answer (62.50%), но не по compute.

Главный вывод: adaptive compute уже полезен, но только до 128 hops; текущий 256-hop bottleneck снова указывает на trajectory ranking / off-policy drift, а не на недостаток ширины beam или параметров. Масштабирование 100M/200M/300M по-прежнему откладывается.

## Следующий шаг

R37: fix the remaining long-horizon trajectory-ranking failure rather than scale parameters. Train an off-policy prefix/trajectory verifier on beam-generated hard negatives, then reuse the R36 uncertainty allocator. Primary target: materially raise 128-hop exact-path selection and obtain non-zero exact-path survival at 256 before any 100M/200M/300M sparse-workspace scaling.

## Артефакты

`r36_metrics.json` — полный calibration/held/extension; `r36_metrics.csv` — компактная таблица; `r36_adaptive_verifier_beam.py` — воспроизводимый runner; `r36_config.json` — frozen protocol; `test_r36_semantics.py` — sanity tests; `r35_base.py` + `r35_verifier_weights.npz` — frozen R35 dependency snapshot.

Примечание: два монолитных запуска runner упёрлись в tool-wall-time и исключены из результатов. Финальные JSON/CSV агрегированы только из полностью завершённых condition/seed chunks с тем же frozen config; незавершённые chunks в метрики не попали.