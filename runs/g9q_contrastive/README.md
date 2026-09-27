# G9Q — query-conditioned contrastive answer supervision

Fresh-data paired mechanism study registered after completed G8C. G8C itself is not repeated.

Hypothesis: for memory-update prompts sharing the same write history but asking different keys, adding a sequence-level pairwise ranking loss that prefers the correct completion over the sibling query's answer will reduce query-insensitive memory retrieval. Control is ordinary answer-only NLL on the identical positive examples. Both arms still execute the same positive/negative scoring forwards; the NLL control multiplies the ranking term by zero, so scoring-path compute is matched as closely as this implementation permits.

Architecture is the same 469,648-parameter causal d-a-d-a hybrid family with true inter-token gated-delta state used by the recent small probes. This remains a small generative mechanism test, not the target 300–800M model and not evidence of Qwen parity.

Training is 800 fixed updates, batch 16, three model-initialization seeds 7901/7902/7903, identical sampler sequence within each paired comparison. Every update contains four complete two-query memory groups plus eight non-memory examples. Contrastive coefficient is 0.30. Fixed final checkpoints only; held sets never select checkpoints or hyperparameters.

Fresh semantic groups are generated before training and are disjoint from all 8,952 canonical groups reconstructed from G8C. Within G9Q, train/validation/held are split by semantic group before rendering. Memory canonicalization deliberately ignores the queried key while preserving each key's chronological write history, so paired query variants never cross splits.

Primary gates: >=5pp mean counterfactual exact gain, >=10 mean both-correct-pair gain out of 200, >=20-pair gain in query-sensitive output changes, positive counterfactual direction in >=2/3 seeds, and mean non-memory IID regression no worse than -2pp. These gates are mechanism evidence only.

Free generation is prompt-only greedy byte decoding with EOS required; gold is not visible to generation/scoring. Report family accuracies, both-correct memory pairs, output-change sensitivity, EOS/truncation, generated lengths, wall time, total/active parameters, raw outputs, checkpoint SHA256, safetensors reload equality and optimizer/RNG next-update replay.
