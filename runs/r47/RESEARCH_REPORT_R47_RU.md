# FlyGraph R47 — joint Bayes protected address belief

**Статус:** REFUTED / MIXED

## Точная гипотеза

Maintaining an explicit joint posterior over protected memory address and reasoning state will outperform trajectory beams by preserving address uncertainty without losing state-address correlations.

## Протокол

- Calibration seeds [4721, 4722]; held [4741, 4742, 4743]; selected likelihood temperature=40.0.
- Main propagates exact joint posterior P(address, reasoning state) over the finite protected state space; latest-write memory semantics are enforced structurally.
- Единственная абляция: factorized P(address)P(state), same observations/transitions, which destroys address↔state correlation.

## Held-out memory/reasoning

| Hops | N | Greedy | R35 fixed32 | **R47 joint Bayes** | Factorized ablation | R47 address top1 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24 | 95.83% | 95.83% | **91.67%** | 79.17% | 94.79% |
| 32 | 18 | 27.78% | 83.33% | **16.67%** | 5.56% | 71.88% |
| 64 | 18 | 16.67% | 83.33% | **33.33%** | 5.56% | 63.11% |
| 128 | 12 | 8.33% | 58.33% | **16.67%** | 8.33% | 47.66% |

## Вердикт

**REFUTED / MIXED**

8-hop Δ vs R35 -4.17 п.п.; 64-hop Δ vs R35 -50.00; 128-hop Δ vs R35 -41.67; 128-hop joint vs factorized +8.33.

## Следующий шаг

R48: diagnose posterior diffusion vs model mismatch by testing protected top-k truncation of the joint Bayes belief (mass-preserving sparse support) against the untruncated joint filter.
