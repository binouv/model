# G9Q — query-conditioned counterfactual contrast

Preregistered mechanistic continuation from the **actual G8C verified-trace final checkpoints**. Two arms branch from the exact same parent model/optimizer state for each ancestral seed. Architecture is unchanged so this isolates the training objective.

The failure targeted here is concrete: G8C often ignored which memory key was queried. Training examples therefore use one write history with two different queried keys whose current values differ. The contrast arm ranks the correct answer sequence above the other key's answer under the same prompt. The control computes the same positive/negative candidate tensors but optimizes only positive NLL.

All primary train/validation/held groups are newly generated and query-invariant at the split level. No G8C held item is used for training. Free generation never receives cases, gold answers, or an executor. Teacher-forced score margins are diagnostics only.

This is a 469,648-parameter synthetic probe, not a new 300–800M model and not evidence of Qwen parity. Fixed step600 endpoints only.
