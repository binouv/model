# R03 — Supervised semantic bottleneck recovers facts from a real frozen Qwen hidden state

**Completed research report · 2026-10-12**  
Code: [probe_and_decode.py](src/probe_and_decode.py). Verified results: [metrics.json](results/metrics.json), [all 48 predictions/method](results/predictions.json). Exact SHA256: [CHECKPOINT_MANIFEST.json](results/CHECKPOINT_MANIFEST.json). Binary checkpoints and optimizer state are stored in [results/](results/). Pre-registered method: [PREREGISTRATION.md](PREREGISTRATION.md).

## Abstract

In R02, an unsupervised text-generation objective applied to the last hidden state of a frozen **Qwen3-0.6B** prompted a small renderer to ignore its input (0/48 exact heldout). Here we test whether the same actual pretrained hidden states represent recoverable semantic facts. Using the *identical SHA-verified cached states*, we train supervised 3×6 fact probes and a compact **semantic bottleneck → latent → GRU** student. A linear probe recovers all three explicit facts on **48/48 unseen combinations**, and **33/48** full triples when verbs are replaced by unseen synonyms. A matched hash-BOW linear probe attains 48/48 IID but only **4/48 OOD**. The dual-style 172,754-parameter latent decoder then generates both short and detailed forms accurately on **48/48 IID** and **30/48 OOD synonym** combinations, vs **5/48 OOD** for the same student fed hash-BOW features. Zeroing/shuffling the inferred latent destroys the text result. The frozen pretrained model does not run again during text realization.

**Interpretation:** the question-final hidden state carries linearly decodable semantic information under these toy constraints; R02's decoder objective failed to retrieve it. The result **does not** show complete frontier answer planning, lossless language-model imitation, long-answer correctness, or real LLM inference speedup.

## Controlled protocol

- Source teacher: the **actual official Qwen/Qwen3-0.6B** revision 167b8104f88905a951069f5f95f9776908da5f68. Entirely frozen in R02. SHA256 of published hidden-state cache: `e0add6315af0ac9f344c90d7339881d8711b49da28576eaae8e5e05055f8ff96`, independently verified in R03.
- Data: 6 subjects, 6 verbs, 6 objects; 168 training triples, 48 unseen complete triples (all individual factors remain observed), plus the same 48 questions with held-out verb synonyms absent from training questions.
- Features: 1024D **final hidden state** of pretrained Qwen after reading question and assistant-start prefix. No gold answer and no required style fed to Qwen.
- Baseline: signed-hash BOW question features of the same 1024D; exactly matched student architecture and steps.
- Experiment 1: direct linear probe and 128-hidden MLP probe with supervised 3×6 slot labels. Label-shuffle control.
- Experiment 2: MLP semantic logits → 3 soft categorical embeddings (each 16D) → 48D latent → tiny style-conditioned recurrent word decoder. Loss = slot classification CE + autoregressive text CE. Train 900 updates, same seed and budget for both sources.
- At evaluation: free-running generation without teacher forcing. Renderer called for both styles with the **same latent**. Zero-latent / shuffled-latent causal controls.
- No Qwen weights copied to repository; only derived hidden cache and 1–2MB **actual student weights plus optimizer/RNG**.

## Main results

| Test (48 new combinations) | Real Qwen feature | Hash-BOW |
|---|---:|---:|
| Linear probe: IID full 3-slot accuracy | 48/48 | 48/48 |
| Linear probe: novel verb-synonym full 3-slot accuracy | **33/48 = 68.75%** | 4/48 = 8.33% |
| MLP probe: novel verb-synonym full triple | 31/48 = 64.58% | 3/48 = 6.25% |
| Full dual-style student: IID both exact | **48/48** | **48/48** |
| Full dual-style student: OOD both exact | **30/48 = 62.5%** | 5/48 = 10.42% |
| Semantic student: OOD full triple | **34/48 = 70.83%** | 7/48 = 14.58% |
| Zeroed latent: paired IID exact | 0/48 | 1/48 |
| Shuffled latent: paired IID exact | 0/48 | 0/48 |
| Shuffled label probe: IID full triple | **0/48** | — |

**Paired OOD comparison:** 4 examples both correct; 26 correct only with Qwen; 1 only with hash-BOW; 17 neither. The exact two-sided discordant-pair binomial/McNemar p-value is approximately **4.17 × 10^-7** on this fixed synthetic test. It measures evidence on **this test distribution and single seed**, not generalization to arbitrary user questions or LLM tasks.

The full student has **172,754** trainable parameters; training took 4.33 seconds (Qwen input features) and 4.35 seconds (BOW input features) on the GitHub Actions CPU. The original teacher forward passes were measured in R02; no new passes are included in this timing.

## Why R02 failed while R03 worked

The cached states have **very high cosine similarity across prompts**: mean pairwise cosine **0.99348**. Their average centered variation is only about **8.07%** of full state L2 norm. This means the large constant/shared component dominates naive vectors, even though individual semantic coordinates carry discriminative signal. Hash-BOW inputs in contrast have pairwise cosine 0.9350 and centered norm ratio 0.2544.

A tiny R02 decoder trained directly on text cross-entropy appears to have converged toward a generic answer and ignored the weak differences between final-prompt teacher states. In R03, separate semantic-label losses force the projection to preserve the animal, verb and object. Then a tiny renderer turns the learned class distributions into either response style.

The fact that a simple **linear** probe gets 48/48 IID and 33/48 OOD supports *accessible fact information in the teacher state*, not merely a fragile nonlinear fitting trick. The shuffled-label control's 0/48 triples further argues against deterministic label leakage through a split bug.

Crucially, the categorical slot labels are engineered supervision. The experimental pipeline provides the semantic schema (three 6-way fields), which would have to be **learned, discovered, or extended to variable-length structured plans** in a general-purpose system. This is not evidence that Qwen's original hidden state already stores exact token strings for an arbitrary answer.

## Precision / speed / limitations

1. **The 48D latent is not a lossless compression of the 1024D state**. It is a supervised representation of a restricted three-field task. The language distribution differs from the official Qwen decoder.
2. Full target Qwen sampling would require autoregressive teacher decoding; that end-to-end matched-quality test is not done here.
3. This is simple semantic copying/composition from explicit question facts, not hidden multi-hop reasoning, coding, actual RU/EN conversation, or factual memory recall.
4. All reported training/evaluation from one fixed, pre-registered seed. The OOD synonyms have different lexical realization but same templated question structure.
5. The teacher CPU encoding cost (R02: ~87.25s / 264 short prompts) still exists. This experiment demonstrates decoder feasibility, not full system throughput versus optimized llama.cpp.
6. Original teacher official weights are not republished. Frozen feature cache SHA and actual small-trained learner tensors, optimizer, RNG and predictions are published. The GitHub Actions verification reloaded checkpoints, recomputed all metrics and passed semantic unit tests.
7. Previously discussed lossless RAM/VRAM weight and KV compression is a *separate and complementary* line; the 48D semantic bottleneck is task-specific and lossy.

## Next scientific progression

- Replace schema-specific triplet supervision with **language-driven plan targets** or structured reasoning traces and compare against text-CoT and tiny end-to-end direct renderer at equal parameters.
- Evaluate real arithmetic/multi-hop questions where the correct answer is **not already directly stated in the input**. A simple hash-BOW baseline and frozen pretrained teacher must both be included.
- Compare final-prompt vector with middle-layer states and learned query pooling of multiple token positions; report additional teacher prefill/memory cost.
- Use confidence-gated multi-slot plans; permit call-backs to Qwen only when uncertainty is high; record actual fallback rate.
- Only after accuracy is credible test CPU DDR4/DDR5 and RTX 3090 with equal output quality, 2K/20K/100K context, TTFT, ITL, total completion time and resident RAM/VRAM.

## Reproduction

```bash
# Using already published and SHA-verified pretrained teacher feature cache
python -m pip install torch numpy
python research/token_delta/r03_semantic_bottleneck/src/probe_and_decode.py \
    --out research/token_delta/r03_semantic_bottleneck/results \
    --probe-steps 700 --joint-steps 900
cd research/token_delta/r03_semantic_bottleneck/src
python -m unittest -v test_semantic_bottleneck
```

**Conclusion:** R03 supports the narrow hypothesis that pretrained Transformer context states can contain reusable semantic representations beyond an immediate next-token instruction. **A trained bottleneck, not a naive small generator, was necessary**. R03 is a genuine positive comparison on a defined task, not a claim of a drop-in LLM accelerator.
