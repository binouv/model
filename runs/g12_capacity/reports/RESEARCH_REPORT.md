# FlyGraph G12 — completed capacity learnability screen

**Verdict: NOT_CONFIRMED_IN_REGISTERED_PROTOCOL.**

Six fixed-step models completed on fresh canonical groups. The paired change was capacity only: **469,648** versus **2,118,928** parameters with the same four-block causal-hybrid pattern, byte259 final-only objective, minibatch sequence and 2,000 updates. This is not compute matched.

Primary arithmetic+code IID accuracy was **0.2083% in both arms**: gain **0.0 pp**, versus the registered +5 pp gate. Overall IID moved only +0.0417 pp; surface +0.0833 pp; extrapolation +0.0833 pp; counterfactual members 0.0 pp. Both-correct counterfactual pairs remained zero. A 4.5x width increase alone therefore did not produce an arithmetic/code learnability regime in this protocol.

The first data build exhausted the old 0..127 canonical space after strict denial of G5S+G7R+G11R groups. Before any optimizer update, the numeric domain was preregistered as 0..511 and extrapolation 512..1023; final split hashes were fixed before training.

After all six step2000 native checkpoints were already written, scoring failed because the copied G5 generate API requires prefix_lens. No held score had succeeded. A scoring-only fix supplied the already-known batch prompt length; weights, prompts, parser, generation policy and gates were unchanged. All six immutable endpoints were then scored.

A process audit also falsely concluded background trainers had stopped because its grep pattern missed their command line. They had not. An attempted resume helper failed an assertion before any optimizer update. Model/optimizer/RNG state was unaffected; text logs were reconstructed from the native step2000 checkpoint logs.

All six final safetensors reload with zero mismatch over 96 replayed generations, and all six optimizer/RNG next-update replay checks pass. Native checkpoints contain model, AdamW, Python, torch and sampler RNG.

Supported batch32 forward+backward FLOPs were about 1.474B small versus 6.601B wide per profiled batch; unsupported operations may be omitted. This is a capacity test, not an equal-FLOP comparison.

Scope remains narrow English synthetic reasoning. G12 does not establish Qwen3.5-4B thinking parity. Taken together, G7R rejects textual trace curriculum as sufficient, G11R rejects naive shared recurrence as sufficient, and G12 rejects width alone as sufficient. The next test changes algorithmic curriculum/order rather than stacking more capacity or recurrence.
