# FlyGraph R50 — terminal-state anchored backward consistency

**Статус:** REFUTED / MIXED

## Точная гипотеза

A delayed reverse-memory consistency pass, anchored by each candidate terminal reasoning state and terminal pointer, can rerank complete protected R35 trajectories better than forward-only R35 and better than the same reverse pass without terminal-state anchoring.

## Протокол

- Forward stage is frozen protected R35 beam32/expand8; no soft posterior propagation.
- After the full forward trajectory set exists, each final candidate is checked by a separate reverse-memory beam under the reversed relation program.
- Reverse paths invert the retrieved operation permutations and must close back to the observed initial reasoning state and noisy initial address.
- Main uses both reverse address closure and terminal-state-conditioned closure probability.
- Единственная абляция uses the identical reverse beam and address closure but marginalizes terminal reasoning state.
- Calibration selected λ main=0.25, ablation=0.25 on seeds [5021, 5022]; held seeds [5041, 5042, 5043].

## Held-out memory/reasoning

| Hops | N | Greedy | R35 fixed32 | **R50 backward anchored** | Address-only ablation | R50 final true path | R50 selected exact path |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24 | 95.83% | 100.00% | **100.00%** | 100.00% | 100.00% | 100.00% |
| 32 | 18 | 33.33% | 100.00% | **100.00%** | 100.00% | 100.00% | 100.00% |
| 64 | 18 | 5.56% | 94.44% | **88.89%** | 94.44% | 94.44% | 88.89% |
| 128 | 12 | 8.33% | 58.33% | **58.33%** | 58.33% | 58.33% | 58.33% |

## Вердикт

**REFUTED / MIXED**

8-hop Δ vs R35 +0.00 п.п.; 64-hop Δ vs R35 -5.56; 128-hop Δ vs R35 +0.00; 128-hop Δ vs ablation +0.00.

## Архитектурное значение

R50 intentionally keeps cognition non-verbal: protected address state, reasoning state, probabilities/scores and an optional later output head. This is compatible with a future multi-rate design in which fast probabilistic decisions can run more frequently than slow reasoning or language generation.

## Следующий шаг

R51: keep protected discrete R35 trajectories, but replace independent reverse retrieval with meet-in-the-middle trajectory agreement: forward and reverse beams must converge on shared midpoint address/state. If that also fails, close bidirectional reranking and move to explicit learned error-correcting address codes before parameter scaling.
