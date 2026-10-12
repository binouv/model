# R02 — Frozen Qwen3-0.6B latent planner, compact text renderer (preregistered)

Date: 2026-10-12. Separate from FlyGraph R-series. **No experiment scores exist at preregistration**.

## Question
Can an actual **pretrained, frozen** Qwen3-0.6B provide one prompt-derived latent state from which a trained small language renderer creates both short and detailed responses **without further Qwen calls**?

## Teacher and privacy
Model: `Qwen/Qwen3-0.6B` revision `167b8104f88905a951069f5f95f9776908da5f68`, official safetensors weights. Only text; no trainable teacher params. Hugging Face download in ephemeral Linux GitHub Actions CPU runner; local sandbox cannot access HF. Record exact resolved commit, model config, tokenizer, package versions, runtime CPU, and save only learner weights (not the 1.2GB copyrighted/redownloadable teacher). This is **not** a weight-distillation license circumvention.

## Data and leakage control
Use 6 subjects × 6 actions × 6 objects (216 fact triples). Heldout 48 disjoint triples and 168 training triples, seed 20261012. Build English prompts as ordinary questions describing the triple in natural text, never include short/detailed target string or requested answer in teacher prompt. The teacher sees the **question only**, and the style is supplied solely to the renderer. Output two short/detailed deterministic surface forms per triple. Evaluate with free-running greedy generation, no teacher forcing or target tokens.
- Holdout A: unseen *combinations*, same lexical verbs as train.
- Holdout B: same heldout combos, but **verb synonyms not present in train prompts**; tests transfer of semantic features.
- Validate no overlapping triples and that every training/heldout concept appears independently.
- These tests are compositional/copying tasks, not broad natural-language QA or reasoning.

## Model and controls
Extract last hidden state of last non-padding user-prompt token from frozen Qwen. Train `Linear(1024→96) / SiLU / Linear(96→48)` planner bridge, renderer GRU with 48D latent and small training-derived vocabulary; renderer should produce both styles from the same latent without a new Qwen call. Freeze the teacher and cache embeddings before training.
Controls:
1. Identical planner-bridge/renderer receiving deterministic hash-BOW question features, no pretrained weights.
2. Evaluate trained Qwen-latent system with zeroed or permuted heldout Z.
3. Train/test multiple styles and report exact tokens + semantic slots, both paired.
4. Time teacher prefill and renderer CPU separately, record model counts and generated token counts; **do not compare raw token counts to optimized llama.cpp or claim 3090 gains**.
Training: same budget, heldout excluded. Fixed seeds and deterministic test.

## Decision criteria
- **Feasibility:** real pretrained Qwen successfully loaded; its nonzero, finite vectors generated from questions; student checkpoint saved as genuine bytes, reloaded and retested; two answer lengths generated without revisiting teacher.
- **Positive transfer:** on OOD paraphrases, Qwen-latent paired semantic accuracy must exceed hash-BOW baseline by >10pp; otherwise classify as unconfirmed/refuted, not success.
- **Deployability:** no claim of improved user-visible generation unless tested versus Qwen teacher-only on real reasoning/knowledge and CPU/RAM/VRAM timing with controlled context lengths.
- Report failure, low accuracy and negative comparisons in full.
- **No unmeasured 10–50× promises.**

## Artifacts
`src/real_qwen_latent.py`, `src/test_real_qwen_latent.py`, `config.json`, `results/metrics.json`, trained student .pt files containing optimizer/RNG, `results/CHECKPOINT_MANIFEST.json`, `REPORT_RU.md`, GitHub Actions workflow. No partial CI run is marked completed. All binary files <90,000,000 bytes, Git push ordinary non-force. 