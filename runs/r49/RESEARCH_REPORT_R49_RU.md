# FlyGraph R49 — discriminative protected joint filter

**Статус:** REFUTED

## Точная гипотеза

Using the frozen R35 discriminative consistency verifier as transition evidence inside an explicit protected address×reasoning-state belief filter will correct the generative-likelihood mismatch and outperform both R35 beam search and the same joint filter driven only by local retrieval likelihood.

## Протокол

- Главный приоритет: cognition / memory / reasoning; сенсорные каналы не изменялись.
- Frozen R35 consistency verifier; protected joint address×reasoning-state belief.
- Transition support: top-8.
- Calibration seeds: `4921, 4922`; held-out seeds: `4941, 4942, 4943`.
- Gamma grid: `0.5, 1.0, 2.0, 4.0`; выбран `gamma=2.0` до held-out.
- Единственная абляция: тот же joint filter, но local retrieval likelihood вместо R35 verifier.
- Oracle labels не используются при inference.

## Completed held-out memory/reasoning

| Hops | N | Greedy | R35 fixed32 | **R49 discr. joint** | Local-likelihood ablation | R49 address top-1 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24 | 95.83% | 100.00% | **100.00%** | 100.00% | 100.00% |
| 32 | 18 | 55.56% | 94.44% | **61.11%** | 88.89% | 88.72% |
| 64 | 18 | 0.00% | 88.89% | **22.22%** | 55.56% | 76.56% |

## Незавершённые данные

Планировавшийся held-out `128-hop` batch был запущен параллельно, но wall-time завершился до появления хотя бы одного законченного episode JSON.  
**0 эпизодов 128-hop включено в результаты.** Эти попытки не используются ни в одной метрике и не выдаются за экспериментальный результат.

## Вердикт

**REFUTED.**

- 8-hop Δ R49 vs R35: `+0.00` п.п.
- 32-hop Δ R49 vs R35: `-33.33` п.п.
- 64-hop Δ R49 vs R35: `-66.67` п.п.
- 64-hop Δ R49 vs local-likelihood ablation: `-33.33` п.п.

Заранее заданный критерий был conjunctive. Он уже строго нарушен на полностью завершённом 64-hop наборе: R49 не только не превосходит R35, но и существенно хуже единственной абляции. Поэтому 128-hop не нужен для определения verdict.

### Что это доказывает

Frozen R35 verifier полезен как **trajectory-level reranker**, но его score нельзя напрямую интерпретировать как локальную Bayes-emission/transition probability внутри размазанного joint posterior. Контекстные verifier-features теряют смысл при posterior-weighted усреднении, и мягкое propagation снова смешивает несовместимые reasoning histories.

Это усиливает архитектурный вывод: protected/discrete memory state следует сохранять в явных траекториях; неопределённость надо удерживать множеством hypotheses, но **не превращать trajectory verifier в локальный soft transition model**.

## Следующий шаг

R50: close the soft forward belief-filter family and test delayed decision via backward consistency. Keep the protected R35 beam forward without soft posterior propagation, then rerank complete surviving trajectories using an independent reverse-memory consistency pass from terminal evidence. This targets global trajectory selection rather than local transition emission.