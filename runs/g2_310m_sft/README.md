# FlyGraph G2-310M — verified-answer SFT

Status: preregistered training in progress; no final quality result is claimed in this commit.

Continuation of the actual G1 step192 BF16 export, upcast to FP32, with a NEW optimizer. This is NOT exact FP32/optimizer continuation of G1: those FP32 archives are not mounted in this session. All 309,993,253 parameters are trainable; the architecture and 8192-token vocabulary are unchanged.

One primary hypothesis: completion-only supervised loss improves free-generated verified RU/EN/code answers over full-sequence language modelling under identical examples, starting weights and update counts. One ablation: full-sequence loss. Both arms receive 256 optimizer updates of four examples with the same family/language bucket sampler. Checkpoints are saved every64 steps. No best-test checkpoint selection.

4096 training cases are independently replayable: arithmetic, conditionals, code traces, list reasoning and mutable-memory lookup. Semantic groups, including operator/query variants and language renderings, are split before rendering. Validation160. Held: IID160, unseen surface80, numerical/length extrapolation80, counterfactual80 (40 pairs). Greedy full-vocabulary generation, max8 new tokens, no external tools. The evaluator never gives answers or cases to the model.

Success requires at least5pp IID exact-answer improvement over both G1 and the matched full-sequence arm, positive combined surface/extrapolation gain and positive both-correct counterfactual-pair gain. Statistical confirmation additionally requires paired sign-test p<.05. Loss reduction alone is not a reasoning success.

The long-term target is a 300–800M generative reasoner surpassing Qwen3.5-4B and approaching Qwen3.8-27B in measured reasoning. Encyclopedic memory and maximum context may be lower, but parameter count, extra inference compute and external tools must be disclosed separately. Neither Qwen model has been run in this experiment; model-card results are not paired baselines.

Publication: source/configuration and completed checkpoints' metadata are pushed incrementally. Large checkpoint bytes are local unless an explicit verified upload receipt says otherwise. A manifest is never a substitute for weights. Existing G1 and historical R-series files are not overwritten.
