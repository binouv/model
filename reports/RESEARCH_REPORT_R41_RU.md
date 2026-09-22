# FlyGraph R41 — horizon-conditioned pairwise disagreement verifier

**Статус:** REFUTED / MIXED  
**Приоритет:** интеллект / reasoning / память. Audio/video не изменялись.

## Точная гипотеза

A horizon-conditioned pairwise verifier trained specifically on real R35-vs-residual disagreement states can decide which trajectory ranking to trust better than static/local rules, improving noisy 64/128-hop reasoning over protected R35 while preserving 8-hop performance.

R40 показал, что one-step counterfactual consistency не различает полезные residual divergence. R41 обучает отдельный pairwise chooser только на фактических состояниях, где protected R35 и residual scorer выбирают разные top trajectory. Inference использует только oracle-free признаки обеих веток и явное conditioning на длину/остаточный горизонт.

## Протокол

- Train seeds: [4111, 4112, 4113]; calibration: [4121, 4122]; held-out: [4141, 4142, 4143].
- Noise p=0.30; beam=32; expand=8; residual beta=0.1.
- Main verifier: pairwise disagreement evidence + explicit horizon interactions.
- Единственная абляция: тот же pairwise dataset/policy/classifier, но без явных horizon features/interactions.

## Held-out memory/reasoning

| Hops | N | Greedy | R35 fixed32 | **R41 pairwise+horizon** | No-horizon ablation | R41 final true path |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24 | 95.83% | 95.83% | **95.83%** | 95.83% | 100.00% |
| 32 | 18 | 33.33% | 100.00% | **100.00%** | 100.00% | 100.00% |
| 64 | 18 | 0.00% | 88.89% | **88.89%** | 83.33% | 88.89% |
| 128 | 12 | 0.00% | 50.00% | **50.00%** | 50.00% | 41.67% |

## Disagreement routing

| Hops | Method | Disagreement/step | Accept residual | Protected choice | Selected exact path |
|---:|---|---:|---:|---:|---:|
| 8 | R41 horizon | 0.00% | 0.00% | 0.00% | 95.83% |
| 8 | No horizon | 0.00% | 0.00% | 0.00% | 95.83% |
| 32 | R41 horizon | 0.35% | 11.11% | 0.00% | 100.00% |
| 32 | No horizon | 0.35% | 0.00% | 11.11% | 100.00% |
| 64 | R41 horizon | 3.99% | 30.54% | 19.46% | 88.89% |
| 64 | No horizon | 2.60% | 10.74% | 39.26% | 83.33% |
| 128 | R41 horizon | 7.68% | 52.05% | 47.95% | 41.67% |
| 128 | No horizon | 6.84% | 25.12% | 74.88% | 41.67% |

## Вердикт

**REFUTED / MIXED** — pairwise disagreement supervision did not satisfy the full long-horizon held-out criterion; trajectory-selection uncertainty remains unresolved enough to block parameter scaling.

8-hop Δ vs greedy: +0.00 п.п.; 64-hop Δ vs R35: +0.00 п.п.; 128-hop Δ vs R35: +0.00 п.п.; 128-hop Δ vs no-horizon: +0.00 п.п.

## Следующий шаг

R42: test a multi-step branch-value verifier trained on disagreement states, predicting short protected rollout survival rather than a binary immediate branch preference; keep memory/retrieval/operators frozen and do not scale parameters yet.
