# R04-H1c — Is the answer computed incrementally, rather than present after prefill?

**Registered: 2026-10-12. Status: hypotheses + software path only, NO RTX 3090 H1c measurements.**

R04 H1 stays `NOT_CONFIRMED_under_registered_gates`; H1b independently diagnoses prompt/format/thinking. H1c tests a DIFFERENT, previously untested hypothesis: a frozen autoregressive Qwen3.5-4B may expose operands before it has internally computed both result digits. Additional trainable **latent recurrence** might compute the result without generating any textual thought tokens. Such a module is a task-specific solver; success DOES NOT imply 4B precomputed the answer.

## Central separation

- `readout` (fixed trained projection of frozen pre-answer hidden state): measures **information accessible without new computation**.
- `latent reasoner` (trainable weight-tied GRUCell, steps T=0,2,8): actively **adds sequential computation**, is not a passive parser.
- `writer` (future separate experiment): Qwen3.5-0.8B receives the head-decided latent **without question, style, length flag or solution**. The writer's own pretrained weights are not proven passive merely by freezing them.
- Optional future `head-LoRA` would MODIFY the original teacher and therefore requires a distinct R05 experiment, not retroactive R04-H1 validation.

## Frozen configuration

- Qwen/Qwen3.5-4B revision `c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97` from existing `../MODEL_LOCK.json`, BF16, no teacher training. Text-only `Qwen3_5ForCausalLM` with 2560 hidden width. Prompt-only final token hidden state, generated before answer tokens. No oracle labels, reasoning traces or future assistant tokens in teacher inputs. Any extra recurrent steps are in a small adapter and their FLOPs/time are counted.
- Frozen 0.8B not invoked in H1c. R04 H2 remains untested, and may not be claimed completed based on H1c.
- GPU target RTX3090 24GiB, Windows/Python3.13, PyTorch2.10+cu128, transformers5.13.0. PEAK reserved <=22GiB. Train recurrent probes separately on already extracted CPU FP32 features so 4B is not resident during probe optimizer steps.

## New disjoint dataset (not the original H1/H1b test)

Generate 1,700 items with seed `20261014`, distributed equally across `(arithmetic,code) × (en,ru)`; each bucket `300 train / 50 val / 75 test`. Exclude **every operand/operation tuple** used in prior H1 (1,700) or H1b (128). The same deterministic 2-digit arithmetic/code-output oracle as H1 is retained for direct comparability; original H1 test is not reused. This is a constrained modular arithmetic diagnostic, NOT general multi-hop reasoning. Commit canonical data and SHA256 before GPU extraction: **canonical SHA256 `ac764eed8a0e06486dee8452219293941e208618fd95cbfa7495395f71184af7`**, JSONL file SHA256 `cf50d791033ffce7d254a3e86d51a4a3d816696b7b9b701f556396fe4ebb9f41`. All 300 holdout test items are scored; no post-hoc success-only filtering.

## Hypotheses and falsifiable gates (locked prior to measurements)

**C0: prefill answer is already easily decodable.** Test trained T=0 probe against T=0 structured and lexical probes. Evidence only if frozen-head T=0 both-digit exact >=40% on all 300 fresh items and >=15 percentage points above best input-only T=0 probe for >=2/3 seeds. Failure is not a denial that information is somewhere else in the model.

**C1: incremental latent computation improves solution.** For **identical recurrent model architecture/parameter count** train at T=0,2,8 with identical 600 steps, AdamW lr0.0015, batch96, seeds (11,22,33). T=8 must reach >=30% exact and beat pretrained-head T=0 by >=10 percentage points on 300 test cases in >=2/3 seeds. Matched structured and hashed-lexical recurrent probes must also be measured; if they match Qwen, improvement is attributed to learning the two-step algorithm in the adapter, not to special precomputed Qwen reasoning. No claim that original 4B itself solved the question is allowed unless direct teacher accuracy demonstrates it.

**C2: adaptive readiness without textual reasoning.** On the T=8 adapter, choose a stopping confidence threshold from {0.55,0.65,0.75,0.85,0.95} on **validation only**. At test, attempt exits after T=0,2,4,8 by checking both-digit softmax confidence (minimum confidence of the two predicted digits); test must be within <=3 percentage points of fixed T=8 exact accuracy while mean actual steps <=4.0; otherwise unconfirmed. All stop rates, accuracy, latency measured per example and over the same population. Calibration by max probability is a proxy, not guaranteed correct-answer detection.

**C3: student merely verbalizes head-chosen answer.** No claim in H1c. After C1, a *separately preregistered* R04-H2b may train a frozen Qwen3.5-0.8B soft-prefix bridge on correct computed-answer plans and head-decided length; it must test writer with zero/shuffled/contradictory plans and no question leakage. Matched-quality speedup and full latency are required before any acceleration claim.

## Causal / scientific checks

1. Run frozen-head feature extraction on all 1700 new questions, pre-answer. Save `features.safetensors`, SHA256 manifest, pinned model revision, tokenizer metadata; neither labels nor text answers enter teacher forward.
2. Probes train on only 1200 examples; val 200 selects ONLY C2 threshold; test 300 measured once per preregistered depth/input/seed, reporting full table including negative results.
3. Frozen-head vs lexical vs structured input controls have same recurrent hidden width and same optimization budget; different input dimensionality affects input projection parameter count, which must be reported.
4. Do not conflate improved supervised adapter accuracy with the model having already calculated the answer. Include train-label shuffle diagnostic on T=8 Qwen as extra negative control.
5. Separate logits/classifier success from ability to produce natural language. Capturing two digit labels in 20 logits is task-specific, and is **not a universal answer representation**.
6. Retain actual checkpoint bytes and optimizer/RNG; each part <90 million bytes with SHA256. Publish only completed, verified run artifacts. Reports may be negative. H1b results remain independent.
7. Do not use original H1 or H1b heldout labels to tune a hypothesis, architecture or threshold; this package was designed after seeing H1 failure and must use its fresh, independently heldout dataset.

## Related original works

- Latent Recurrent Thoughts (Sept 2026): https://arxiv.org/abs/2609.01117 ; implementation https://github.com/czl-david/latent-recurrent-thoughts
- LOTUS Looped Transformers (June 2026): https://arxiv.org/abs/2606.31779 ; implementation https://github.com/yingfan-bot/lotus
- Thinking States (Feb 2026): https://arxiv.org/abs/2602.08332
- RecurTrace / adaptive loop-depth (Sept 2026): https://arxiv.org/abs/2609.03379
- Think Before You Speak / pause tokens (2023): https://arxiv.org/abs/2310.02226

This package is an independent, lightweight diagnostic implementation. It does not reimplement or claim the results of the cited papers.