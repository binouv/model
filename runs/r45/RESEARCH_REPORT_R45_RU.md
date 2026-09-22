# FlyGraph R45 — posterior address-state propagation

**Статус:** REFUTED / MIXED

## Точная гипотеза

Explicitly propagating posterior mass over compact (protected address signature, reasoning state) hypothesis groups, with stratified particle allocation, will preserve ambiguous memory trajectories better than global top-k scoring and improve 64/128-hop noisy reasoning.

## Протокол

- Calibration seeds [4521, 4522]; held [4541, 4542, 4543]; no learned parameters.
- Predictive group signature: (reasoning_state, inferred next-address subject id).
- Main uses logsumexp group posterior mass, deterministic posterior quota resampling, and equal-weight particles whose total weight preserves group mass.
- Единственная абляция: identical grouping/resampling but group evidence is max-path score, so no evidence aggregation.
- Calibrated per-group particle cap=2 before held-out.

## Held-out memory/reasoning

| Hops | N | Greedy | R35 fixed32 | **R45 posterior** | Group-max ablation | R45 final true path |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24 | 100.00% | 100.00% | **100.00%** | 100.00% | 100.00% |
| 32 | 18 | 38.89% | 94.44% | **88.89%** | 22.22% | 94.44% |
| 64 | 18 | 33.33% | 83.33% | **83.33%** | 16.67% | 83.33% |
| 128 | 12 | 0.00% | 33.33% | **25.00%** | 8.33% | 33.33% |

## Вердикт

**REFUTED / MIXED** — posterior grouping/resampling does not satisfy the full held-out criterion; address uncertainty needs richer state than a single predicted next-subject signature.

8-hop Δ vs R35 +0.00 п.п.; 64-hop Δ vs R35 +0.00; 128-hop Δ vs R35 -8.33; 128-hop Δ vs ablation +16.67.

## Следующий шаг

R46: enrich the uncertainty state itself, not the scorer: propagate a top-k address distribution/signature per particle (rather than a single canonical next subject) and merge by distributional overlap; compare against R45 single-address posterior.
