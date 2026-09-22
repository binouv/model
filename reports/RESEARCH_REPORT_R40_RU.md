# FlyGraph R40 — counterfactual rollback for residual trajectory corrections

**Статус:** REFUTED / MIXED  
**Приоритет:** cognition / reasoning / memory. Audio/video не затрагивались.

## Точная гипотеза

When the protected R35 cumulative score and off-policy residual ranking disagree, accept the residual-preferred trajectory only if a one-step counterfactual protected look-ahead predicts at least comparable future consistency; otherwise rollback to the protected beam. This should improve 128-hop ranking without sacrificing 8/64-hop performance.

R38/R39 показали, что ограничение величины residual может убрать вред, но не отвечает на вопрос, когда residual действительно заслуживает доверия. R40 сравнивает protected R35 ranking и residual ranking на каждом шаге. При расхождении residual-предпочтённая ветка принимается только если одноступенчатый counterfactual look-ahead через frozen R35 scorer даёт ей не хуже future protected score с откалиброванным запасом. Иначе beam откатывается к protected score и накопленный residual сбрасывается.

## Протокол

- Frozen R35 consistency verifier + frozen R37 off-policy residual; новых обучаемых параметров нет.
- Noise `p=0.30`, beam=32, expand=8, residual beta=0.10.
- Calibration seeds: 4021/4022, по 2 эпизода/condition; held-out seeds: 4041/4042/4043.
- Delta grid: -0.50/-0.25/0/0.25/0.50/1.00; выбран `delta=1.00`.
- Единственная абляция: agreement-only rollback — при любом disagreement немедленно откатываться к R35 без look-ahead override.
- Таймаутнутый монолитный процесс не включён. OOD32/OOD64 после таймаута объединённого процесса были заново завершены отдельными standalone runs.

## Held-out memory/reasoning

| Hops | N | Greedy | R35 fixed32 | **R40 counterfactual** | Agreement-only |
|---:|---:|---:|---:|---:|---:|
| 8 | 18 | 94.44% | 100.00% | **100.00%** | 100.00% |
| 32 | 18 | 44.44% | 88.89% | **88.89%** | 88.89% |
| 64 | 18 | 16.67% | 94.44% | **94.44%** | 94.44% |
| 128 | 12 | 0.00% | 50.00% | **50.00%** | 50.00% |

## Trajectory metrics

| Hops | Method | Path survival | Final true path | Selected exact path | Disagreement | Rollback | CF accept | Lookahead evals |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | R35 fixed32 | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0 |
| 8 | R40 CF rollback | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0 |
| 8 | Agreement-only | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0 |
| 32 | R35 fixed32 | 99.48% | 94.44% | 88.89% | 0.00% | 0.00% | 0.00% | 0 |
| 32 | R40 CF rollback | 99.48% | 94.44% | 88.89% | 0.87% | 0.87% | 0.00% | 4 |
| 32 | Agreement-only | 99.48% | 94.44% | 88.89% | 0.87% | 0.87% | 0.00% | 0 |
| 64 | R35 fixed32 | 96.44% | 94.44% | 94.44% | 0.00% | 0.00% | 0.00% | 0 |
| 64 | R40 CF rollback | 96.44% | 94.44% | 94.44% | 2.08% | 2.08% | 0.00% | 20 |
| 64 | Agreement-only | 96.44% | 94.44% | 94.44% | 2.08% | 2.08% | 0.00% | 0 |
| 128 | R35 fixed32 | 78.78% | 75.00% | 50.00% | 0.00% | 0.00% | 0.00% | 0 |
| 128 | R40 CF rollback | 78.78% | 75.00% | 50.00% | 6.97% | 6.25% | 13.40% | 141 |
| 128 | Agreement-only | 78.78% | 75.00% | 50.00% | 6.97% | 6.97% | 0.00% | 0 |

## Абляция и вывод

На 128 hops R40 видел global disagreement примерно на **6.97%** шагов и разрешал residual override примерно в **13.40%** таких disagreement. Несмотря на это, answer accuracy, final true-path retention и selected exact path остались **точно такими же**, как у R35 fixed32 и agreement-only ablation. На 32/64 hops counterfactual override почти никогда не принимался.

**REFUTED / MIXED.** 8-hop delta vs greedy = +5.56 п.п.; 64-hop delta vs R35 = +0.00 п.п.; 128-hop delta vs R35 = +0.00 п.п.; 128-hop delta vs agreement-only = +0.00 п.п.

Гипотеза не подтверждена: one-step protected look-ahead полезен как safety/rollback mechanism, но не как достаточно информативный selector полезных residual corrections. Он нейтрализует drift, однако не улучшает ranking поверх R35.

## Следующий шаг

Do not scale 100M/200M/300M yet. R41 should train a horizon-conditioned pairwise preference verifier specifically on actual R35-vs-residual disagreement states, predicting which branch wins under future protected consistency. Keep protected/discrete memory and reusable operators frozen so the test isolates trajectory selection rather than capacity.
