# R04-H1 — preregistered computation extraction from a frozen Qwen3.5-4B

Date: **2026-10-12**. This document extends, **without relaxing**, the existing [R04 PREREGISTRATION](../PREREGISTRATION.md). It defines a concrete reproducible H1 computation benchmark before seeing any R04-H1 model result. No real RTX 3090 H1 measurement was performed while authoring this protocol. The previously completed RTX 3090 4B/0.8B GPU **engineering smoke** is not H1 proof.

## Scientific question

Does the **last pre-answer token** hidden state of an unmodified, frozen Qwen3.5-4B contain enough *already-computed* information to linearly/nonlinearly decode both digits of a programmatically verifiable **two-step arithmetic or Python-output result**, without first generating an answer or a chain-of-thought token?

Success would show that these output facts are decodable **for this restricted distribution** before language decoding. It would **not** show the entire forthcoming response is in one vector, that the frozen 0.8B can realize arbitrary plans, or that 99% of FLOPs can be eliminated.

## Fixed model / hardware

- Frozen Qwen/Qwen3.5-4B BF16, revision `c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97`; read source `../MODEL_LOCK.json` at runtime. Transformers 5.13.0, PyTorch 2.10.0+cu128, RTX 3090 24 GiB; peak limit <22 GiB. Text-only `Qwen3_5ForCausalLM`, no training of teacher weights.
- Context: single-turn user question, chat template `enable_thinking=False`, one assistant-start marker; inputs **never include** the oracle answer, any teacher-generated answer, or any chain-of-thought. Hidden `hidden_states[16]`, `[24]`, `[32]` at **last prompt position**, captured in one forward call, `output_hidden_states=True`, `use_cache=False`.
- Heads 16/24 are fixed a priori; `last` is layer 32. Frozen features retained in **BF16 safetensors** with SHA256. Readout trains in FP32. Only small diagnostic **probes** train: R04 H2's writer/bridge must not be modified by H1.

## Dataset (hard-locked)

`h1/src/h1_dataset.py` deterministically generates **1,700 unique** question tuples from seed **20261012**, and writes exact `.jsonl` / SHA manifests. Groups are `(operator,a,b,c)` with `a,b,c ∈ 0..19`, operation from `add`, `weighted`; a group appears in **exactly one** split and language/domain bucket. No paraphrase of the same underlying problem crosses splits.

Stratification: four equally sized buckets, `(arithmetic,code) × (en,ru)`. Each bucket has **300 train / 50 validation / 75 test**. Overall **1,200 / 200 / 300**. The sample `answer` is two computed digits `p q`.

- **add:** `p=(a+b)%10`; `q=(p+c)%10`.
- **weighted:** `p=(2a+b)%10`; `q=(p+2c)%10`.
- **arithmetic:** formula is described in text. **code:** fixed Python source is shown, with assignments, modulo operations and `print(x,z)`; the code-based oracle is checked against actual Python execution on all generated cases. All prompts demand the two digits **separated by whitespace**, no explanations or other text.
- The output is a pair of computed facts, rather than an R03-style direct copy of subject/object strings. Exact success requires **both** values; report each position separately. Also report tokenizer histogram proving whether target requires >=2 tokens. The result is still a toy benchmark of very short outputs, not multi-sentence hidden planning.

The written question is only a deterministic function of the *operands/template*; oracle values never enter the teacher's prompt string. Tests assert group uniqueness, oracle/interpreter agreement, reproducibility and byte tampering detection.

## Frozen teacher baselines

1. Direct **4B greedy generation** on all 300 test examples, `max_new_tokens=12`; parse strict two decimal digits separated by whitespace, any extra text or invalid answer counts wrong. Teacher accuracy is over all 300 (including refusals, bad formatting, early stop). **Never filter the test for correct teacher answers**; report conditional correctness separately.
2. Direct **0.8B greedy generation** on the same 300 test questions, identical prompt/output rules, as a standalone model. The writer is *not* shown head latent; this is not H2.
3. Test answer tokenizer length histogram and separate arithmetic/code and RU/EN accuracy; report EOS truncation and invalid outputs. No posthoc retries, no chain-of-thought or reranking.

## Readouts and control groups

`h1/src/h1_probes.py` trains independent small models predicting two categorical digits (`2 × 10` logits) with **cross entropy**. Main features are `layer16`, `layer24`, `last`, all width 2560. Two exact architectures per feature:

- `linear`: `LayerNorm(2560) → Linear(2560,20)`.
- `mlp`: `LayerNorm(2560) → Linear(2560,160) → GELU → Linear(160,20)`.

Comparisons:

- Strong **hash lexical** control: position-sensitive unigram/bigram/trigram hash from **the same question**, dimension 2560, without any oracle answer.
- Stronger additional **structured input** control: exact three operand one-hots, operation, domain and language (66 dimensions), with the corresponding same probe architectures. It receives the question's parsed inputs, **not** computed answers, testing whether simple readouts could themselves solve the program. This control is explicitly reported rather than hidden.
- **Permuted-label control:** same last-layer MLP on a deterministic shuffled mapping from training inputs to training answer labels. Test/validation labels remain untouched.

Training is frozen before H1: seeds **11,22,33**; **600 optimizer steps**, batch **96**, AdamW learning rate **0.0015**, weight decay 0.0001, no early stopping. The same settings apply to every candidate and control. No test values enter training or model selection. Save full native **weights + optimizer + RNG** per candidate per seed, all <90,000,000 bytes each, with SHA256; cache interrupted/sharded forward results separately from complete science runs.

**Selection:** use ONLY the **mean validation exact accuracy across the 3 seeds** to choose the best teacher `(layer, architecture)` and best non-teacher input `(hash/structured, architecture)`; tie-break deterministic lexicographic ordering. Report validation summaries for all candidates. Score only chosen configurations on test, 3 seeds each, plus chosen shuffled-label control. Raw per-example outputs and per-digit accuracy are mandatory.

## Falsifiable success gates — unchanged from R04

H1 is CONFIRMED **under this fixed protocol** only when all hold:

1. Frozen Qwen3.5-4B greedy test accuracy on all programmatic questions **≥80%**.
2. Chosen frozen-head probe exact **both-digit accuracy ≥70%** on all 300 test problems for **at least 2 of 3 seeds**.
3. For those same ≥2 seeds, head probe exceeds the **strongest selected non-teacher input control by ≥15 percentage points** and the permuted-label control by **≥20 points**.
4. All baseline runs (including 0.8B), required per-example reports, source hashes and checkpoint reloads complete. Missing runs => **INCOMPLETE**, not a successful test.

A gate failure is a **negative result under the registered setup**, not proof that no meaningful future information exists in other hidden states or with different training.

## Internal validity requirements

- Exact full SHAs, tokenizer/revisions, package versions, GPU and data hashes in manifests.
- At least 95% of test answer strings must tokenize to >=2 tokens for any claim involving **future tokens beyond first**; otherwise the interpretation is "short-answer probing only".
- Report per-digit vs both-digit accuracies. Strong first-digit performance with weak second digit does not show that the second token was ready.
- Report all 300 examples (success/failure) and strict teacher formatting failures, even if the teacher has lower accuracy than the threshold.
- H1's frozen teacher prefill uses BF16 and model inference; downstream stored BF16 features preserve that type, but this is **not the lossless VRAM/KV codec research line**.
- Run smoke / GPU pilot separately and label it ENGINEERING. Never publish synthetic/random fixture metrics as if real model measurements.

## Source and status

Code: `h1/src/h1_dataset.py`, `h1/src/h1_pilot.py`, `h1/src/h1_extract.py`, `h1/src/h1_probes.py`; tests `h1/tests/test_h1.py`. A CPU-only authoring test and a fake-feature end-to-end software-path test validate mechanics, not H1. The actual user's RTX 3090 has not yet executed this H1 implementation.