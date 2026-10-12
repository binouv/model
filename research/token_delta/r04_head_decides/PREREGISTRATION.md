# R04 — Head Decides / Student Renders (preregistration)

**Registered 2026-10-12, before any R04 RTX 3090 scientific measurements.** Status: **H1–H4 NOT TESTED**. GPU hardware isn't available in the authoring environment. The separate `gpu_smoke.py` is an *engineering validation* only and must not be reported as a successful scientific experiment.

## Contract / central hypothesis

Frozen official **Qwen3.5-4B** (head) receives the user question and **chooses the content, reasoning outcome, level of detail, and when to finish**. Frozen official **Qwen3.5-0.8B** (writer) sees **only** a sequence of K learned continuous prefix embeddings from the head, plus a fixed assistant-start token string. It receives **no user question**, **no explicit short/long flag**, **no answer text at inference**, and **no answer length or style label**. Its own next-token sampling is a mechanical realization, not proof that it adds zero information. **Only the bridge is trained** (no LoRA or changes to either base model).

Prompt-to-head hidden extraction is **one forward pass**, with no teacher-generated answer/CoT tokens included in the state. During training, head-generated answer text may be used as *offline targets* but NEVER supplied to the head's input or to the writer at test time. The cost of producing these offline targets belongs to training-data preparation; at inference, target answers are unavailable. Keep these accounting boundaries explicit.

R03 on Qwen3-0.6B proved a **supervised three-slot copying interface** on synthetic English facts; its 30/48 synonym-generalization result neither established new reasoning nor any end-to-end speedup. R04 explicitly tests nontrivial computed answers and model-appropriate response length.

## Locked setup

- `MODEL_LOCK.json` lists immutable full Hugging Face SHAs and text-only Qwen3_5ForCausalLM. Verify resolved SHA and dimensions at runtime. FP32 bridge, BF16 frozen base models; frozen prefill features cached as CPU FP32 with per-file SHA256.
- RTX 3090 24 GiB; start `batch=1`, `prompt<=256 tokens`, `max_output<=96`, `K=4`, no quantization, CUDA PyTorch, Python 3.13. Stop on CUDA OOM or reserved peak >=22 GiB in the first smoke.
- Hold out complete questions; do not mix paraphrases of the same underlying problem between train/test. Deterministic seed `20261012`; record software, environment and GPU clock/thermal state.
- Train/validation/test split predeclared: **1,200 / 200 / 300** prompts generated deterministically in RU/EN from independent parameter tuples. Test contains distinct problem instances, not just changed punctuation. Target domains: two-step arithmetic, exact numeric constraints, short code-output prediction, explanation requiring two linked facts. Store both programmatic oracle where possible and head's sampled/greedy response. Report teacher-correct and all-test metrics; no test-based filtering of failures. For non-exact explanation outputs use blind semantic rubric and publish examples.
- Teacher head's length decision can reflect the full user request, but the writer must never receive that request except through K vectors. Evaluation contains predeclared tasks likely to prompt either short or detailed answers, not an externally imposed `style_id` on the writer.

## H1 — An answer is decodable before answer tokens are generated

**Claim:** After head prefill but *before* generation of answer tokens, answer-relevant hidden information exists beyond trivial lexical identity. Probe `last`, and as a preregistered ablation the intermediate `layers[16]` and `layers[24]` on identical test instances; inspect one final-token state first, then cached pooled slots separately.

**Protocol:** Freeze 4B completely. Train a linear probe and a 2-layer MLP (same budget as corresponding hash-BOW probes) to predict oracle-checked arithmetic/code answers. Test 300 disjoint problems; use 3 seeds for readout training (11,22,33) and report mean + raw per-seed scores. Controls: label permutation, hash-BOW, last position of prompt with numbers randomly swapped, and standalone 0.8B fed the original task.

**Confirm only if:** teacher-only correctness on programmatic subset >=80%; trained frozen-head probe >=70% exact *all-test* on that subset AND >=15 percentage points over strongest lexical probe AND >=20 points over shuffled-label control, in >=2/3 seeds. **Refuted under this protocol** if any gate fails. Failure does not prove no semantic latent exists in other layers or trained states.

## H2 — Frozen 0.8B can render a decided answer from head-only latent

**Claim:** A learned bridge `2560 -> K*1024` can convey answers and *implicitly* their necessary level of detail, without passing the question/style to the frozen 0.8B.

**Protocol:** Use K=4 primary and K=1/8/16 ablations (same train examples, steps, optimizer, microbatch). Optimize only bridge against frozen-head **offline target answers on train**, never teacher answers from test. Compare with frozen 0.8B conditioned on independent input question, an otherwise identical bridge fed hash-BOW features (same model capacity), zero/shuffled learned prefixes, and direct head generation. Score oracle fact correctness, exactness for deterministic outputs, paired human-blind semantic equivalence for prose.

**Confirm only if:** >=75% correct of *all* deterministic test tasks AND at least 80% of head-correct tasks AND >=15 points above the hash-BOW bridge baseline AND >=30 points above shuffled-latent control; 95% Wilson confidence intervals and seed variance are reported. Else **not confirmed**. No partial label/style tokens or target text may be passed to the writer during evaluation. At this stage no speed claim.

## H3 — Head determines length / stopping, not the writer

**Claim:** The same writer realizes both concise and elaborate head decisions solely because the **latent plan differs**. Writer receives no length token, response-length class or requested style. Head may be asked naturally by the user to clarify/explain or just compute; its intent is encoded only in the head state.

**Protocol:** For the fixed validation/test prompts, generate head-only reference responses with `enable_thinking=False` (separately report thinking-enabled ablation). Stratify for analysis by predeclared response-length bands: short <=24 tokens, detailed >=80 tokens, according to head's output. Full test set and per-band denominators always disclosed. Train length behavior only from head responses in the training split. Test EOS/stop, length ratio, content semantic match. Do NOT pass a Boolean `is_detailed` to the writer.

**Confirm only if:** >=85% of test responses land in the *same* predeclared length band as the head, median writer/head token-length ratio in [0.8,1.25] separately per band, and semantic accuracy loses <=10 percentage points vs teacher-only. At least 30 examples per band required for interpretable band-level claims; if not enough, H3 is **inconclusive**, not posthoc rebalanced.

## H4 — Net speedup on RTX 3090 at matched quality

**Claim:** For sufficiently long answers, `4B prefill + one bridge + 0.8B decode` can be faster than `4B prefill + 4B decode`, when the teacher is not called each output step.

**Protocol:** Keep all stages GPU-resident if memory permits; include full head prefill time, bridge latency and writer autoregressive generation time; do **not** count only student tokens/s. Matched BF16/hardware, batch1, warm up >=3, >=30 distinct prompts, 3 repeat runs. Explicit context buckets 256 / 2048 / 20000 tokens, output targets 32 / 128 / 512 (report truncation/early stop rather than fake fixed-length decoding). Compare quality, actual EOS, TTFT, total wall-clock, peak VRAM. `torch.cuda.synchronize()` around timings. If optimized llama.cpp Q4 baseline is used, mark as secondary due to differing quantization.

**Confirm only if:** median end-to-end speedup >=1.25x *on the same answered-task distribution*, quality drop <=5 percentage points (semantic / exact as appropriate), and 95th-percentile latency does not worsen by >10%; no OOM. Otherwise **not confirmed**. CPU DDR4 may be investigated in an entirely separate benchmark; do not extrapolate 3090 results to RAM.

## Publication and anti-leakage requirements

1. Scientific results appear only after completed run; engineering smoke metrics never count as a win.
2. Commit source/config/tests/outputs/complete trainable bridge weights and optimizer+RNG states; each binary part <=90,000,000 bytes, with actual bytes and SHA256 manifest.
3. Never claim the writer is a passive printer in a mechanistic sense unless ablation removes answer-solving contribution. Its 0.8B frozen weights still contain learned knowledge and may reconstruct missing facts.
4. Publish raw per-example predictions of head, writer, lexical, unconditioned and shuffled baselines; keep negative results.
5. Exact reproduction means offline cache/pinned revisions, seeds, tokenizer IDs, full prompt templates and model-output accounting.
6. Any output exceeding the H4 quality gate but not speed gate is a *model-compression research result*, not a speed improvement.
7. Do not modify FlyGraph R-series checkpoint training or publish incomplete model artifacts as completed research.