# FlyGraph R39 — horizon-aware cumulative residual trust region

**Статус:** REFUTED/MIXED  
**Приоритет:** cognition / memory / reasoning. Audio/video не затрагивались.

## Точная гипотеза

Bounding the cumulative off-policy residual by a horizon-aware trajectory trust region will preserve a useful medium-horizon ranking correction (strictly improve 64-hop accuracy over protected R35) while preventing residual drift from degrading the protected R35 cumulative score at 128 hops.

## Протокол

- Frozen R35 cumulative verifier и frozen R37 off-policy residual; новых обучаемых параметров нет.
- Main: residual не может изменить trajectory score больше horizon-aware cumulative budget.
- Budget: `B(T)=2.0 * min(1, 64/max(64,T))^3`: B64=2.0, B128=0.25, B256=0.03125.
- Calibration: seeds 3921/3922, 2 episodes/seed/condition; gamma={1,2,3}. Все три дали одинаковый calibration objective, поэтому выбран gamma=3 как заранее заданный tie-break на более сильное long-horizon ограничение.
- Held-out: fresh seeds 3941/3942/3943, 4 episodes/seed/condition.
- Единственная абляция: такой же cumulative cap=2.0, но без horizon-dependent shrink.
- Незавершённые/таймаутнутые монолитные сетки и предварительные held seeds 3931–3933 исключены из итоговых метрик.

## Held-out memory/reasoning

| Hops | N | Greedy | R35 fixed32 | **R39 horizon trust** | Constant-cap ablation |
|---:|---:|---:|---:|---:|---:|
| 8 | 12 | 91.67% | 91.67% | **91.67%** | 91.67% |
| 32 | 12 | 50.00% | 100.00% | **100.00%** | 100.00% |
| 64 | 12 | 0.00% | 75.00% | **75.00%** | 75.00% |
| 128 | 12 | 8.33% | 33.33% | **33.33%** | 33.33% |

## Trajectory / memory metrics

| Hops | Method | Path survival | Final true path | Selected exact path | Cap activation | Budget |
|---:|---|---:|---:|---:|---:|---:|
| 8 | R35 fixed32 | 100.00% | 100.00% | 91.67% | 0.00% | 0.000 |
| 8 | R39 horizon | 100.00% | 100.00% | 91.67% | 7.39% | 2.000 |
| 8 | Constant cap | 100.00% | 100.00% | 91.67% | 7.39% | 2.000 |
| 32 | R35 fixed32 | 100.00% | 100.00% | 100.00% | 0.00% | 0.000 |
| 32 | R39 horizon | 100.00% | 100.00% | 100.00% | 59.60% | 2.000 |
| 32 | Constant cap | 100.00% | 100.00% | 100.00% | 59.60% | 2.000 |
| 64 | R35 fixed32 | 100.00% | 100.00% | 75.00% | 0.00% | 0.000 |
| 64 | R39 horizon | 100.00% | 100.00% | 75.00% | 81.40% | 2.000 |
| 64 | Constant cap | 100.00% | 100.00% | 75.00% | 81.40% | 2.000 |
| 128 | R35 fixed32 | 58.20% | 41.67% | 33.33% | 0.00% | 0.000 |
| 128 | R39 horizon | 58.20% | 41.67% | 33.33% | 97.51% | 0.250 |
| 128 | Constant cap | 58.20% | 41.67% | 33.33% | 82.62% | 2.000 |

## Вердикт

**REFUTED/MIXED.** 8-hop Δ vs greedy = +0.00 п.п.; 64-hop Δ vs R35 = +0.00 п.п.; 128-hop Δ vs R35 = +0.00 п.п.; horizon-aware vs constant-cap at 128 = +0.00 п.п.

Trust region устранил наблюдавшийся ранее destructive residual drift: на свежем held set R39 не хуже R35 на 128 hops. Но одновременно он не дал требуемого **строгого** улучшения на 64 hops. Более того, horizon-aware и constant-cap абляция дали одинаковые answer/path-selection метрики. Значит статический cumulative budget работает в основном как ограничитель residual authority, а не как доказанный length-aware selector полезных corrections.

На 128 hops main budget уменьшен в 8 раз относительно 64 hops (2.0 → 0.25), но это не изменило итоговый ranking относительно constant cap=2.0. Следовательно, следующий bottleneck — не величина residual сама по себе, а определение **когда** конкретную correction можно безопасно принять.

## 256-hop extension

Не запускался: основной критерий R39 не выполнен. Нельзя тратить следующий цикл на глубину/масштабирование до исправления selection policy.

## Следующий шаг

R40 should test rollback/counterfactual trajectory agreement rather than another static residual budget: permit a correction only when the protected R35 and residual scorer disagree transiently but a short look-ahead restores cross-step consistency. Do not start 100M/200M/300M scaling yet.
