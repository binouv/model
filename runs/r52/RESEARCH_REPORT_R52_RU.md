# FlyGraph R52 — learned linear ECC protected address codebook

**Статус:** REFUTED / MIXED

## Точная гипотеза

A learned high-minimum-distance protected address code, without increasing the 96-bit address width or changing the frozen reasoning/verifier stack, will reduce noisy memory ambiguity enough to improve long-horizon 64/128-hop reasoning. It must also beat an equal-capacity random linear-code control, so any gain is attributable to distance optimization rather than merely using a fixed code family.

## Протокол

- Главный приоритет: cognition / memory / reasoning; audio/video/body не затрагивались.
- Protected address width оставлен **ровно 96 бит**: это не parameter/workspace scaling.
- Main: линейный бинарный `[96,9]` код на 512 codewords; generator обучался только на геометрической ECC-цели. Train seeds `[5211, 5212, 5213]`.
- Selected code minimum Hamming distance: **41**; p05: **42.0**.
- Единственная абляция: random `[96,9]` generator той же ёмкости/ширины; min distance **35**, p05 **40.0**.
- Entity→codeword mapping заново случайно переставляется в каждом эпизоде, поэтому фиксированную семантику entity memorization использовать нельзя.
- Frozen R35 verifier, beam=32, expand=8. Внутри каждого эпизода path/program/updates/operations и **точные noise flip masks** совпадают между raw/main/ablation.

## Completed held-out memory/reasoning

| Hops | N | Raw96 greedy | Raw96 R35 | **R52 optimized ECC** | Random-linear ablation | R52 path survival | R52 final true path |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 36 | 88.89% | 97.22% | **100.00%** | 97.22% | 100.00% | 100.00% |
| 32 | 24 | 37.50% | 100.00% | **91.67%** | 95.83% | 97.01% | 95.83% |
| 64 | 24 | 8.33% | 66.67% | **70.83%** | 66.67% | 80.27% | 70.83% |
| 128 | 18 | 5.56% | 44.44% | **44.44%** | 38.89% | 55.12% | 38.89% |

## Semantic validation

Semantic mismatches: **0**. Per condition: `{'id8': 0, 'ood32': 0, 'ood64': 0, 'ood128': 0}`.

## Вердикт

**REFUTED / MIXED.**

- 8-hop Δ main vs raw96 R35: **+2.78 п.п.**
- 32-hop Δ: **-8.33 п.п.**
- 64-hop Δ: **+4.17 п.п.**
- 128-hop Δ: **+0.00 п.п.**
- 64-hop main vs random-linear ablation: **+4.17 п.п.**
- 128-hop main vs random-linear ablation: **+5.56 п.п.**

Оптимизация code distance действительно дала локальный сигнал: main выиграл у random-linear контроля на 64 и 128 hops и улучшил matched raw96 на 64 hops. Но ключевой long-horizon критерий не выполнен: **128-hop answer не вырос относительно raw96 R35**. Значит, одной статической ECC-геометрии недостаточно. Bottleneck смещается от codebook geometry к **декодированию неопределённости**: при p=.30 надо сохранять несколько syndrome/codeword hypotheses, а не пытаться сделать один статический код настолько хорошим, чтобы hard selection всегда был верным.

## Следующий шаг

R53: syndrome-aware multi-hypothesis decoding. Keep the R52 optimized protected code and frozen R35 scorer, but decode a noisy pointer to a small discrete set of syndrome/codeword hypotheses with calibrated posterior weights instead of one hard address. Compare against hard nearest-code decoding as the single ablation. Do not scale 100M/200M/300M yet because 128-hop gain over raw R35 was not established.
