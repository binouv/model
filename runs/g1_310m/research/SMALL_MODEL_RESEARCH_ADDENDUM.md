# Additional official research for 300M–800M FlyGraph

Reviewed 2026-09-25. These are design/data lessons, NOT new trained FlyGraph results.

## MobileLLM

Paper: https://arxiv.org/abs/2402.14905
Official code: https://github.com/facebookresearch/MobileLLM
Deep-and-thin architecture, embedding sharing, GQA and immediate block sharing are directly relevant to sub-billion models. Compare width/depth under equal actual FLOPs, not just unique parameters. Weight sharing does not make repeated forward passes free. The repository has FAIR Noncommercial Research License; source files and weights were not copied into G1.

## MobileLLM-R1

Official recipe: https://github.com/facebookresearch/MobileLLM-R1
Paper: https://arxiv.org/abs/2509.24945
The official 360M recipe stages pretraining, KL-distillation midtraining, general SFT and reasoning SFT. This is not simply reasoning-RL from random initialization. The teacher/student tokenizer compatibility in their recipe matters: direct token-level KL cannot be applied naively between our8K tokenizer and Qwen's vocabulary. Sequence-level verified teacher examples are the simpler initial route for different tokenizers. FAIRNC restrictions apply to actual code/weight derivatives; no Meta checkpoint or source file was imported here.

## TinyStories

Paper: https://arxiv.org/abs/2305.07759
Controlled synthetic narratives can teach small models coherent simple language. For FlyGraph, use this as motivation for varied RU/EN prose and unseen narrative/fact combinations, not as proof of broad Qwen-level factual/coding capability.

## Priority after the actual G1 result

G1 trained309,993,253parameters on only98,304presentations and solved0/21synthetic prompts. Its within-token graph readout barely changes NLL. This is too little training to estimate an architectural ceiling. Prioritize substantial natural RU/EN/code data, independently verified teacher solutions, a tokenizer designed for that mixture, and meaningful inter-token state; then test gated attention/MTP/recurrent-depth changes in matched experiments. Proposed348/546/794M configurations remain UNTRAINED designs.

The isolated gate implementation uses original headwise2*sigmoid gating with zero initialization, not Qwen's exact elementwise implementation. Six correctness tests passed; no quality improvement was measured. The new memory curriculum passed1000replay/invariance/rejection checks but has NOT trained G1.
