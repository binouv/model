# FlyGraph R46 — distributional address-belief propagation

**Статус:** REFUTED / MIXED

## Точная гипотеза

Propagating a compact top-k posterior over the next protected memory address, and merging particles only when those address distributions overlap, will preserve ambiguity better than R45 single-address grouping and improve 64/128-hop reasoning.

## Протокол

- Calibration seeds [4621, 4622]; held [4641, 4642, 4643]; no learned parameters.
- Top-k address belief K=4; selected overlap threshold=0.35, group cap=2.
- Main merges only same-state particles whose compact address beliefs overlap; group mass is logsumexp.
- Единственная абляция: R45 single canonical next-address posterior grouping on the exact same episodes.

## Held-out memory/reasoning

| Hops | N | Greedy | R35 fixed32 | **R46 distributional** | R45 single-address | R46 final true path |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24 | 91.67% | 95.83% | **95.83%** | 95.83% | 100.00% |
| 32 | 18 | 38.89% | 94.44% | **94.44%** | 94.44% | 100.00% |
| 64 | 18 | 16.67% | 77.78% | **77.78%** | 77.78% | 83.33% |
| 128 | 12 | 16.67% | 41.67% | **41.67%** | 41.67% | 41.67% |

## Вердикт

**REFUTED / MIXED**

8-hop Δ vs R35 +0.00 п.п.; 64-hop Δ vs R35 +0.00; 128-hop Δ vs R35 +0.00; 128-hop Δ vs R45 +0.00.

## Следующий шаг

R47: replace heuristic merging/resampling with an explicit Bayes-style protected address belief filter that propagates address posterior mass without collapsing it into trajectory particles; compare against R35 fixed beam.
