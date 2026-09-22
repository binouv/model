# FlyGraph R48 — sparse protected joint belief

**Статус:** REFUTED / MIXED

## Точная гипотеза

Mass-preserving top-k truncation of the protected joint address×reasoning posterior will prevent harmful soft posterior diffusion and outperform the untruncated joint filter at long horizons.

## Протокол

- Fixed R47 likelihood temperature=40.0; calibration K on [4821, 4822]; selected K=128; held=[4841, 4842, 4843].
- Main keeps only the top-K joint address×state cells after each pointer update, renormalizing retained probability mass.
- Единственная абляция: untruncated R47 joint filter.

## Held-out memory/reasoning

| Hops | N | Greedy | R35 fixed32 | **R48 sparse joint** | Untruncated R47 | R48 addr top1 | Mass kept |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24 | 95.83% | 100.00% | **91.67%** | 91.67% | 95.31% | 99.97% |
| 32 | 18 | 44.44% | 77.78% | **27.78%** | 27.78% | 74.13% | 94.51% |
| 64 | 18 | 22.22% | 83.33% | **33.33%** | 33.33% | 65.71% | 91.37% |
| 128 | 12 | 0.00% | 50.00% | **0.00%** | 0.00% | 55.60% | 85.84% |

## Вердикт

**REFUTED / MIXED**

8-hop Δ vs R35 -8.33 п.п.; 64-hop Δ vs R35 -50.00; 128-hop Δ vs R35 -50.00; 128-hop Δ vs untruncated +0.00.

## Следующий шаг

R49: replace generative Hamming likelihood with the frozen R35 discriminative verifier as the belief transition/emission score, keeping explicit joint state; test whether model mismatch rather than diffusion is dominant.
