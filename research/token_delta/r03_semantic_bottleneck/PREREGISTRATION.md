# R03 — Are semantic facts linearly readable from real Qwen prompt latents?

2026-10-12, **preregistered prior to run**. Depends on **verified R02 official Qwen3-0.6B hidden states** and R02 held-out splits. Does not claim to improve Qwen.

## Motivation / R02 negative result
R02: last-prompt Qwen state → unsupervised answer renderer scored 0/48 exact on IID and OOD heldout; matched hash-BOW baseline 48/48 IID and 3/48 paired OOD. Teacher inference: 87.25s for 264 prompts (CPU), saved all 1024-D states. This does not tell whether facts are missing from latent or text-decoder optimization failed.

## H1: supervised fact probe (failure threshold)
Train `Linear(1024,18)` and `LayerNorm→MLP(1024,128,18)` on exactly 168 **train** prompts to recover three 6-way categories: actor/action/object. Test on 48 heldout *combinations* (same lexicon), plus 48 heldout prompts with synonymous verbs never present in train. Compare *same capacity/budget* on real pretrained Qwen latent vs 1024-D hashed BOW. Report per-category accuracy, exact-triple rate and train-vs-test gap. If <70% joint triple accuracy on IID, do not claim real pretrained hidden state sufficiently supports tiny answer generation.

## H2: semantic supervision for compact renderer
Try a small `semantic bottleneck`:
`Qwen hidden→MLP→3×6 semantic class logits→soft 3-slot embeddings→48D latent→tiny autoregressive GRU`.
Train on 168 question-answer pairs in **both** short and detailed forms with factual cross-entropy plus text cross-entropy. At test no labels; project Qwen feature once and generate both lengths without further Qwen calls. Compare identical architecture/compute budget on hashed BOW. Include zero- and shuffle-latent interventions. Separately report slot accuracies and end-to-end exact text. No per-test adaptive hyperparameter selection.

## Integrity
Student sees target tokens only during training, never during teacher extraction or evaluation. OOD verbs held out from train prompts. All metrics and **negative outcomes** published; split and target grammar exactly R02. Checkpoints must include optimizer, RNG, source identities, SHA256, test predictions and reloaded metrics. Don't cherry-pick samples or call template copying free-form language reasoning.

## Hardware and limitations
GitHub Actions Ubuntu CPU (no local HF access), pretrained teacher from R02 cached feature file with verified SHA256, no trainable Qwen parameters and no teacher re-run. Any R03 improvement is on tiny constrained English copying tasks; no evidence yet of speedup vs real model, 20K contexts, RU/EN code/math or independent frontier-scale generation.