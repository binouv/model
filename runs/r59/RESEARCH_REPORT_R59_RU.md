# FlyGraph R59 — distilled shallow branch value

**Статус: CONFIRMED**

## Точная гипотеза

A compact branch-value student can distill R58 one-step future consistency from current candidate/ancestry signals, preserving long-horizon reasoning while eliminating explicit lookahead expansion.

## Обучение

- Завершённое обучение: **32412 teacher samples**, 34 features, seeds `5911/5912/5913`.
- Student MLP: `34→48→24→1`; iterations **170**; train corr **0.8009**, RMSE **0.8226**.
- Separate calibration: corr64 **0.5299**, corr128 **0.6049**.

## Held-out memory/reasoning

| Hops | N | **R59 student** | Exact R58 | R59 final true path | Exact final true path | Edge-eval ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 12 | **100.00%** | 100.00% | 100.00% | 100.00% | **0.399×** |
| 64 | 12 | **100.00%** | 100.00% | 100.00% | 100.00% | **0.394×** |
| 128 | 6 | **83.33%** | 83.33% | 83.33% | 83.33% | **0.390×** |

## Вердикт

**CONFIRMED.** Student matched exact R58 answer on all three fresh held conditions. 128-hop compute ratio is **0.390×**, ~2.57× fewer edge evaluations.

128-hop generation loss/episode: R59 `0.167`, exact R58 `0.167`; pruning loss/episode: both `0.000`.

Semantic mismatches: **0**. Partial/timed-out aggregates excluded.

## Архитектурный вывод

R58 future-value signal is compressible into current branch/ancestry state: explicit one-step tree expansion is not required on this held distribution. A learned value head can replace a large fraction of deliberative search while preserving protected discrete memory trajectories.

## Следующий шаг

R60: selective exact fallback on student boundary ambiguity/disagreement; R59 pure student is the single ablation. Target R58 quality with <=1.5× R59 compute.
