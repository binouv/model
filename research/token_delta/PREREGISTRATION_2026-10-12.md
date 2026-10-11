# Latent Answer / Token-Delta: preregistration (2026-10-12)

Status: **hypotheses only; no training results at preregistration**. Independent side research; does not modify FlyGraph R-series checkpoints.

## Core hypothesis
An expensive planner encodes a task's semantic answer into one compact latent once. A much smaller autoregressive decoder realizes either a short or detailed text from exactly the same latent **without revisiting the planner**. The two text styles must express identical underlying facts. This does NOT assert that a standard pretrained LLM has fully completed future outputs in its final hidden state.

## Experiment A: future-state probe
Train tiny GRU next-token model on (a) repeated 4-token motifs and (b) independent uniformly random tokens. Freeze it. Learn a linear head mapping its final context hidden state to 8 future token predictions. Same training budget across tasks. Holdout seed disjoint. Predictions are scored against ground truth by future horizon and greedy prefix. Negative control (iid randomness) must remain near 1/16 accuracy. Results do not demonstrate natural-language planning.

## Experiment B: latent plan -> variable-length answer
Generate all 8x8x8 agent-action-object combinations, with disjoint train/test combinations but shared compositional vocabulary. Planner: trainable small embedding/MLP encoder, producing *one* style-independent continuous vector from the input triples. Renderer: separately trainable tiny GRU (text decoder), conditioned on the fixed latent and the requested style. Training targets are deterministic, different short and detailed sentences with the same facts. Decode greedily to EOS without teacher forcing; heldout combinations never seen jointly during training.
- B1 paired short/detailed exact-match on heldout combinations.
- B2 per-slot semantic consistency independent of surface style.
- B3 ablation: zeroed/shuffled latent; reported as causal test.
- B4 count parameters and actual CPU latency for planner once + decoder; compare properly, do not present toy tokens/s as LLM speed.
- B5 save checkpoint state dicts, optimizer state or mark weights-only, config, metrics, samples, test result, and SHA256.
- Proposed pass gate B1 >=90% *paired* exact match; fail otherwise. Predictive gate is not a result.

## Evaluation / falsification
Full sequence exact match, slot fidelity, EOS correctness, detailed-vs-short consistency, shuffled-latent ablation. Test no target tokens leak into teacher state, train vs heldout combinations isolated, random seed fixed. Any speed claim requires matched real-model baseline. No identity to frontier-model generation distribution is claimed. Speculative verification is separate and can be distribution-exact only if implemented correctly.

## Hardware
CPU PyTorch sandbox, no NVIDIA GPU or pretrained Qwen checkpoints available. Models are deliberately very small, synthetic language. Checkpoint bytes must be published to establish completion.

## Related work to compare (not our results)
- Chen et al., *Latent-Guided Reasoning*, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/6958e9d0f5a76d54ff97da8c45f4d52e-Abstract-Conference.html
- Goel et al., *Embedding-Space Probing*, ICML 2026: https://proceedings.mlr.press/v306/goel26a.html
- Xu et al., *How Far Ahead Do LLMs Plan?*, ICML 2026: https://proceedings.mlr.press/v306/xu26cb.html
- Lin et al., *SEED*, 2026: https://arxiv.org/abs/2609.36590
- Hao et al., *Coconut*, 2025: https://github.com/facebookresearch/coconut
- Liu et al., *Speculative Decoding: Performance or Illusion?*, MLSys 2026: https://proceedings.mlsys.org/paper_files/paper/2026/hash/554e056fe2b6d9fd27ffcd3367ae1267-Abstract-Conference.html
