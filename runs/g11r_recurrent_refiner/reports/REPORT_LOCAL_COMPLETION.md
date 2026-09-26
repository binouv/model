# FlyGraph G11R — shared recurrent latent-refinement screen

Completed locally. Registered hypothesis NOT confirmed.

Six fixed-step 2000-update models were trained on fresh canonical groups disjoint from reconstructed G5S and G7R groups. Both arms have exactly 580432 trainable parameters. Main executes one shared causal refiner block four times; control executes the same parameterized refiner once. The four-pass arm is intentionally higher-compute, not compute-matched.

Primary arithmetic+code IID accuracy was 0.625% in both arms: gain 0.0 pp versus registered +3 pp. Overall IID was 1.167% vs 1.083% (+0.083 pp). Memory+list was 1.875% vs 1.458% (+0.417 pp). Surface was equal; extrapolation was zero in both; counterfactual members regressed by 0.5 pp. Both-correct counterfactual pairs remained zero.

Thus naive shared recursion/deeper effective computation does not solve the arithmetic/code bottleneck at this scale. Combined with G7R, textual trace curriculum and simple shared recurrence are both rejected as sufficient mechanisms.

All six final safetensors were reloaded; 96 generated cases matched exactly. Native model/AdamW/Python/torch/sampler RNG checkpoints exist every500 steps and all six next-update replay checks passed. Actual bytes are preserved locally; this GitHub connector exposes repository writes but no release-asset upload action, so no remote binary release is claimed.

The architecture idea was inspired only at the mechanism level by Tiny Recursive Models (arXiv:2510.04871; official SamsungSAILMontreal/TinyRecursiveModels, MIT). No code, weights, data, or pretrained base were imported.

Next: stop adding recurrence at sub-million scale. Run a fresh capacity/data screen focused on arithmetic and code trace (about0.5M vs about2M under the same objective/data), then move only mechanisms that survive into the 300–800M generative backbone.
