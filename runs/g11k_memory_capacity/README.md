# G11K — parameter-matched associative-memory capacity screen

Registered before training. This follows completed G8C and two local negative objective pilots; it does not repeat G3/G5S/G8C.

Hypothesis: at fixed total/active parameters, reallocating capacity from the feed-forward sublayer into the gated-delta key space improves query-conditioned memory retrieval.

Paired architectures, both exactly 107,344 parameters:
- baseline_k8_ff96: dim48, 4 heads, hidden96, delta key_dim8.
- memory_k20_ff80: dim48, 4 heads, hidden80, delta key_dim20.

Both keep the same 4-block d-a-d-a causal hybrid, tokenizer, optimizer, update count and sampler. Three initialization seeds 7971/7972/7973; 480 updates; batch8 formed from four complete query-pairs; fixed endpoint only.

Fresh semantic groups are generated before training and exclude all saved G9P/G10R groups. Train/calibration/held/held-OOD are split by write-history group before rendering. Gold is never visible to free generation.

Primary gates: positive held exact accuracy, positive both-correct pair count, nonnegative query-sensitivity change, and no truncation regression. This is a narrow architecture mechanism test only; it cannot establish Qwen parity or justify 300–800M scaling by itself.
