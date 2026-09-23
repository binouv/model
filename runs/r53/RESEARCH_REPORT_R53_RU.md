# FlyGraph R53 — syndrome-aware discrete multi-hypothesis ECC decoder

**Статус:** REFUTED / MIXED

## Точная гипотеза

Keeping several discrete ECC codeword hypotheses for each noisy protected pointer, rather than forcing hard nearest-code cleanup, will preserve the true address under p=.30 and improve long-horizon 64/128-hop reasoning when the frozen R35 verifier selects among the resulting concrete trajectories.

## Протокол

- Главный приоритет: cognition / reasoning / memory. Audio/video/body не менялись.
- Frozen R52 address representation: binary linear `[96,9]` ECC, 512 codewords, minimum Hamming distance **41**.
- Frozen R35 trajectory verifier: beam **32**, edge expand **8**.
- Main: exact BSC decoder at `p=.30` retains **top-4 discrete codeword hypotheses** per noisy pointer. Addresses and reasoning states are never averaged.
- Concrete edge trajectories are scored by the frozen R35 verifier on the original noisy pointer, plus a small decoder branch prior `beta=0.10`.
- Единственная абляция: identical pipeline with **hard top-1 ECC decode** at every step.
- Held-out seeds: `5351/5352/5353`. Completed episode counts: 18 / 12 / 12 / 6 for 8/32/64/128 hops.
- An exploratory calibration grid timed out and was discarded; it did not select any hyperparameter. The final held-out run uses the fixed implementation constants `top_k=4`, `beta=.10`.

## Completed held-out memory/reasoning

| Hops | N | R52 ECC + R35 | **R53 top-4 decoder** | Hard-decode ablation | True address in R53 top-4 | R53 path survival |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 18 | 94.44% | **94.44%** | 72.22% | 91.67% | 100.00% |
| 32 | 12 | 100.00% | **100.00%** | 16.67% | 94.27% | 100.00% |
| 64 | 12 | 83.33% | **58.33%** | 0.00% | 94.53% | 74.09% |
| 128 | 6 | 16.67% | **0.00%** | 16.67% | 93.10% | 20.83% |

## Semantic validation

`0` semantic mismatches across all completed held-out episodes: `{'id8': 0, 'ood32': 0, 'ood64': 0, 'ood128': 0}`.

The first monolithic held-out launcher timed out after completing the 8/32/64 files. Its unfinished 128-hop work is excluded. The final 128-hop table above is built only from **six separately completed standalone episodes** (two per held seed).

## Вердикт

**REFUTED / MIXED.**

- 8-hop Δ R53 vs R52: **+0.00 п.п.**
- 64-hop Δ R53 vs R52: **-25.00 п.п.**
- 128-hop Δ R53 vs R52: **-16.67 п.п.**
- 64-hop Δ R53 vs hard decode: **+58.33 п.п.**
- 128-hop Δ R53 vs hard decode: **-16.67 п.п.**

Hard nearest-code cleanup is strongly rejected: at 64 hops its answer accuracy falls to 0.00%, while the true address is top-1 only 82.29% of steps. Keeping four discrete decoder hypotheses raises true-address coverage to 94.53% at 64 hops and 93.10% at 128 hops.

But **coverage is not the same as usable cognition**. Always branching top-4 reduces 64-hop answer from 83.33% to 58.33%, and 128-hop from 16.67% to 0.00%. The true address is usually present among decoder hypotheses, yet competing branches consume the fixed trajectory budget and the correct full path disappears earlier.

### Architectural conclusion

R52 showed that better static ECC geometry helps. R53 now shows that **hard correction is too destructive, but indiscriminate multi-hypothesis expansion is also wrong**. The useful object is a discrete uncertainty set, but compute must be allocated selectively. This is consistent with the protected-state principle: retain alternatives without soft averaging, and spend extra cognition only when the decoder is actually uncertain.

## Следующий step

**R54: adaptive uncertainty-triggered ECC branching.** Preserve ordinary R52/R35 behavior on confident steps and open discrete top-k ECC branches only when decoder posterior margin/entropy indicates ambiguity. Use a separately calibrated trigger threshold; one ablation is always-on R53 top-4 branching. No 100M/200M/300M scaling yet.