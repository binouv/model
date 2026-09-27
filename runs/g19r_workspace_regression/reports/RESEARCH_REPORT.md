# FlyGraph G19R report — 2026-09-27

G19R tested whether the negative G18P result was partly caused by sparse 384-way auxiliary classification. A new self-contained 470,020-parameter byte-level gated recurrent generative probe was registered before local training. Three parameter-matched arms were trained from identical per-seed initialization and exact minibatch schedules: final-only, sorted same-multiset (bag) state regression, and chronological ordered state regression. Two fresh seeds (9201, 9202), 80 updates each.

All six endpoint trainings completed. All six safetensors reloads reproduced greedy generations on the audit sample with zero mismatches. Deterministic reserved-next-update replay matched for every endpoint. Real model, optimizer and RNG bytes are present in the local bundle with SHA256 hashes.

Ordered regression improved mean arithmetic+code IID strict exact from 3.90625% to 4.6875% (+0.78125 pp) and beat the bag arm by the same +0.78125 pp. This passes the order-vs-bag gate (>=0.5 pp) but misses the >=1.0 pp ordered-vs-final gate. One of two seeds improved. Chronological workspace MAE improved strongly versus bag (6.477 vs 7.544) but answer-token NLL worsened versus final-only by +0.01349. Memory+list changed by -0.78125 pp, within bound. Surface exact tied final-only; extrapolation remained 0% for every arm.

Verdict: **NOT_CONFIRMED_IN_REGISTERED_PILOT**. There is a small chronology-specific signal, but not enough evidence to scale this objective. This is not a natural reasoning/code benchmark and is not comparable to Qwen3.5-4B thinking.

Important audit limitation: the generated manifest observed a small number of cross-split semantic collisions (train 5, IID 3, surface 2 under the run's canonicalizer). That further weakens generalization claims and must be fixed in the next data generator.

The next experiment should preserve the chronology signal while reducing answer-loss interference and must generate collision-free semantic splits before rendering.
