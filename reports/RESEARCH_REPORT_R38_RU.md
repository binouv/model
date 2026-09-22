# FlyGraph R38 — protected cumulative score + frozen off-policy residual

**Статус:** REFUTED/MIXED  
**Приоритет:** cognition / memory / reasoning. Аудио/видео не затрагивались.

## Точная гипотеза

A frozen off-policy verifier used only as a confidence-gated residual on top of the protected cumulative R35 trajectory score can correct hard ranking errors without destroying long-horizon evidence aggregation.

R38 не переобучает verifier: используется фактически завершённый R37 off-policy model. Это изолирует эффект scoring architecture.

## Протокол

- Calibration: seed 3821, 2 episodes/condition, grid β={0.10,0.20}, gate={0.70,0.85}.
- Selected: β=0.10, gate=0.70.
- Held-out: seeds 3831/3832/3833, 4 episodes per seed per condition; p=0.30; beam32, expand8.
- Main: cumulative R35 trajectory score + confidence-gated residual.
- Единственная абляция: тот же residual без confidence gate.
- Два монолитных запуска упёрлись в wall-time и полностью исключены.

## Held-out memory/reasoning

| Hops | N | Greedy | R35 fixed32 | **R38 gated residual** | Ungated ablation |
|---:|---:|---:|---:|---:|---:|
| 8 | 12 | 91.67% | 100.00% | **100.00%** | 100.00% |
| 32 | 12 | 66.67% | 91.67% | **91.67%** | 91.67% |
| 64 | 12 | 16.67% | 75.00% | **83.33%** | 83.33% |
| 128 | 12 | 0.00% | 75.00% | **58.33%** | 58.33% |

## Trajectory metrics

| Hops | Method | Path survival | Final true path | Selected exact path | Gate rate |
|---:|---|---:|---:|---:|---:|
| 8 | R35 fixed32 | 100.00% | 100.00% | 100.00% | 0.00% |
| 8 | R38 gated | 100.00% | 100.00% | 100.00% | 99.52% |
| 8 | Ungated | 100.00% | 100.00% | 100.00% | 100.00% |
| 32 | R35 fixed32 | 100.00% | 100.00% | 91.67% | 0.00% |
| 32 | R38 gated | 100.00% | 100.00% | 91.67% | 95.71% |
| 32 | Ungated | 100.00% | 100.00% | 91.67% | 100.00% |
| 64 | R35 fixed32 | 86.46% | 83.33% | 75.00% | 0.00% |
| 64 | R38 gated | 92.84% | 91.67% | 83.33% | 94.65% |
| 64 | Ungated | 92.84% | 91.67% | 83.33% | 100.00% |
| 128 | R35 fixed32 | 67.77% | 58.33% | 58.33% | 0.00% |
| 128 | R38 gated | 65.36% | 50.00% | 50.00% | 94.59% |
| 128 | Ungated | 65.36% | 50.00% | 50.00% | 100.00% |

## Вердикт

**REFUTED/MIXED**. 8-hop Δ vs greedy +8.33 pp; 64-hop Δ vs R35 fixed32 +8.33 pp; 128-hop Δ -16.67 pp.

Residual correction действительно помогает на 64 hops, но на 128 hops ухудшает protected cumulative baseline. Confidence gating почти не отличился от ungated ablation на этом held set: текущая confidence measure открывает residual слишком часто и не распознаёт long-horizon harmful corrections.

## Следующий шаг

R39 должен проверять step/trajectory-length-aware residual budget: ограничивать суммарное влияние correction на длинной траектории (например trust-region/cumulative residual cap), а не gate каждый edge независимо. Масштабирование 100M/200M/300M остаётся отложенным до стабилизации 128-hop retrieval/ranking.