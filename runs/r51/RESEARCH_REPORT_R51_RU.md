# FlyGraph R51 — meet-in-the-middle protected midpoint agreement

**Статус:** REFUTED

## Точная гипотеза

Forward and reverse protected trajectory hypotheses that are both plausible locally can be disambiguated by requiring them to meet at a shared midpoint address AND reasoning state; this midpoint joint agreement will rerank R35 complete trajectories better than forward-only R35 and better than midpoint address agreement alone.

## Протокол

- Frozen protected R35 beam32/expand8 generates complete discrete trajectories; no soft forward posterior propagation.
- For each final candidate, a reverse beam traverses only the second half of the relation program.
- Main score is reverse posterior evidence that the reverse hypothesis meets the candidate forward hypothesis at the same midpoint address AND reconstructed reasoning state.
- Единственная абляция marginalizes midpoint reasoning state and scores address agreement only.
- Calibration seeds [5121,5122], held seeds [5141,5142,5143]; selected λ main=0.25, ablation=0.25.

## Held-out memory/reasoning

| Hops | N | Greedy | R35 fixed32 | **R51 midpoint joint** | Address-only ablation | R51 final true path | R51 selected exact path |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24 | 95.83% | 95.83% | **95.83%** | 100.00% | 100.00% | 95.83% |
| 32 | 18 | 44.44% | 94.44% | **94.44%** | 94.44% | 100.00% | 94.44% |
| 64 | 18 | 5.56% | 88.89% | **88.89%** | 88.89% | 88.89% | 88.89% |
| 128 | 12 | 8.33% | 25.00% | **25.00%** | 25.00% | 25.00% | 25.00% |

## Вердикт

**REFUTED**

8-hop Δ vs R35 +0.00 п.п.; 64-hop Δ vs R35 +0.00; 128-hop Δ vs R35 +0.00; 128-hop Δ vs ablation +0.00.

## Следующий шаг

R52: close the bidirectional-reranking family. Test explicit learned error-correcting protected address representations under the same p=.30 mutable-memory benchmark, while keeping R35 trajectory scoring frozen; do not scale workspace parameters yet.
