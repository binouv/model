# R02 — Frozen pretrained Qwen3-0.6B as a one-vector answer planner: a negative controlled result

**Experimental report · 2026-10-12 · completed, reproducible, binary weights published**  
Repo: [binouv/model](https://github.com/binouv/model/tree/main/research/token_delta/r02_qwen_teacher)  
Sources: [Code](src/real_qwen_latent.py), [Tests](src/test_real_qwen_latent.py), [preregistered hypothesis](PREREGISTRATION.md), [metrics](results/metrics.json), [all 48 predictions per method](results/predictions.json), [binary SHA manifest](results/CHECKPOINT_MANIFEST.json).  
Teacher: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) revision 167b8104f88905a951069f5f95f9776908da5f68.

## Abstract

The hypothesis under test is that a pretrained autoregressive teacher can provide a compact semantic plan at the end of prompt prefill, and that a small independently trained renderer can realize answers in different styles without running the teacher autoregressively. A frozen Qwen3-0.6B supplied 1024-dimensional final-layer hidden states for 264 natural-language scene descriptions; a 139,648-parameter learner projected this to one 48D vector and rendered either a short or detailed sentence. A matched-parameter learner using ordinary hashed bag-of-words features was the control. On 48 unseen triples the Qwen-latent learner attained **0/48 exact** for both styles, versus **48/48 exact** for hash-BOW. On heldout synonyms the Qwen-latent learner again scored 0/48 pairs; hash-BOW attained 3/48 pairs. The negative outcome falsifies this **specific readout/training protocol** as an adequate solution. It does **not** falsify richer multi-slot latents, layer-specific probes, a trained teacher, or adaptive refinement.

## Experimental design

The dataset is a deliberately limited English composition task: 6 subjects × 6 verbs × 6 objects = 216 fact triples. The test set consists of 48 unseen complete combinations, but every individual factor appears in training. The teacher gets a question describing the fact triple in natural language, without an answer or instruction about whether the response will be short or detailed. One extra test condition changes verbs to synonyms never encountered in training prompts. There are **168 distinct training prompts, 48 IID-composition tests, and 48 OOD-paraphrase tests**.

Sample question: "Read the following scene carefully. Near the river, the hare hides the stone. What happened? Give the main facts." Short target: "the hare hides the stone." Detailed target: "in the quiet garden, the hare carefully hides the stone near the river today."

The Qwen teacher is loaded from **real official pretrained safetensors**; weights are never updated. Only a frozen final hidden vector for each formatted prompt is extracted (output_hidden_states=True). The 1024-dimensional vector undergoes LayerNorm, Linear(1024,96), SiLU, Linear(96,48), Tanh. The renderer is a style-conditioned GRUCell with small token embedding; total trainable model = **139,648 parameters**. Its decoder *still emits words autoregressively*. The baseline replaces Qwen features with deterministic signed hash-BOW (1024D) and uses exactly the same trainable bridge, renderer, initialization seed and training budget (800 steps, batch 48).

Qwen inference, training, and testing were performed on Ubuntu GitHub Actions **CPU** with torch 2.10.0+cpu, Python 3.12.15 and FP32. Teacher prefill took 87.25 s over 264 questions (9521 input tokens), or ~0.33 s per question under the measured batch and CPU conditions. Weight loading was separately 18.60 s. Learner training took 8.69 s for Qwen features and 8.57 s for hash-BOW features.

## Results, full heldout evaluations

| 48 independently held-out triples | Qwen latent learner | Matched hash-BOW learner |
|---|---:|---:|
| IID short exact | **0/48** | **48/48** |
| IID detailed exact | **0/48** | **48/48** |
| IID paired exact | **0/48** | **48/48** |
| IID short semantic slot mean | 18.06% | 100% |
| IID detailed semantic slot mean | 18.75% | 100% |
| OOD verb synonym paired exact | **0/48** | **3/48** |
| OOD short semantic slot mean | 18.06% | 70.14% |
| OOD detailed semantic slot mean | 18.75% | 71.53% |
| Zero latent paired exact | 0/48 | 0/48 |
| Shuffled latent paired exact | 0/48 | 0/48 |

Observation: most outputs of the Qwen-latent learner were effectively constant across distinct factual questions (e.g., predicting "the fox moves the ring" for different animals and actions), whereas the BOW learner reconstructed the actual question triples in the IID setting. That is a diagnostic sign of *feature/readout collapse* at the chosen single final state and training objective.

The small renderer alone is inexpensive: the measured *single-example, 17 decoding-step* call was ~1.47 milliseconds, bridge ~0.07 milliseconds. This is a tiny constrained-vocabulary recurrent network; these timings cannot establish a speedup over large-model generation, nor apples-to-apples equality of output quality. In particular the CPU teacher still needs a prompt prefill, and the 0% quality renders an alleged speedup meaningless.

## Interpretation and threats to validity

1. This experiment directly contradicts the assertion that a one-vector Qwen final-prompt hidden state is automatically a sufficiently accessible complete answer representation **for a tiny decoder trained under this setup**.
2. The hidden state may still encode information not linearly accessible, or meaningful information may reside at earlier layers, earlier token positions, or multiple states. The failure could also be optimization-related. The experiment does not distinguish these explanations.
3. The baseline task is **copying/combining explicit facts from the question**, not spontaneous factual QA or reasoning. The BOW control's IID success does not prove general intelligence.
4. Synonyms in OOD were disjoint from training questions; BOW performance declined as expected. Yet the Qwen-latent learner did not exhibit beneficial semantic transfer, invalidating our preregistered transfer-positive criterion.
5. The teacher did not generate a target sample and no equal-quality end-to-end teacher baseline is measured. The architecture is **not distribution-exact** with the pretrained Qwen model.
6. The prefill context is short; no 20K/100K token benchmark, no RTX 3090, and no quantitative RAM bandwidth instrumentation.
7. The published checkpoint files contain trainable learner state, optimizer state and RNG. The 1.2GB official teacher weights are **referenced by pinned Hugging Face revision, not copied into this repo**. Cached frozen teacher features and their SHA256 are published to rerun the small learner without the teacher download.

## Next falsifiable experiment (R03)

Using the **already saved exact Qwen feature bytes**, train supervised linear and MLP probes to recover the three separate semantic slots (subject, verb, object). Compare against BOW, do train-label shuffle control, measure cosine geometry and feature variation, then train a student with an explicit supervised semantic bottleneck before the language renderer. Keep the same 48 heldout examples and synonym tests. This distinguishes whether the problem lies in the semantic representation or the unsupervised student objective.

## Broader research context

- [Future Lens (2023)](https://arxiv.org/abs/2311.04897): hidden states can signal future tokens, not necessarily a complete finished answer.
- [Do language models plan ahead for future tokens? (2024)](https://arxiv.org/abs/2404.00859): distinguishing pre-caching from breadcrumbs is relevant to the mechanistic hypothesis.
- [Implicit Reasoning survey (2025)](https://arxiv.org/abs/2509.02350): architectures for continuous latent reasoning and internal computation.
- [Latent-Guided Reasoning, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6958e9d0f5a76d54ff97da8c45f4d52e-Abstract-Conference.html): direct large-thinker/small-executor precedent.
- [Embedding-Space Probing, ICML 2026](https://proceedings.mlr.press/v306/goel26a.html) and [SEED (September 2026)](https://arxiv.org/abs/2609.36590): candidate future-token prediction and fast drafts are distinct from unverified lossless semantic answer transfer.

**Conclusion:** R02 succeeded at genuine pretrained-teacher integration, independent reproducibility, and binary publication. It **failed the accuracy hypothesis** for the specific 1×48D latent learned from the final Qwen prompt state; this failure is explicitly retained rather than hidden or relabelled as a success.
