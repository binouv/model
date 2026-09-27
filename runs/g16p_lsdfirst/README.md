# FlyGraph G16 serialization screen

G16R was preregistered as a three-seed, 600-update paired comparison of conventional most-significant-digit-first output versus least-significant-digit-first output. Its first local execution timed out at step 295 of the first arm and wrote no completed checkpoint, so it is explicitly not a result.

G16P is a separately preregistered finite pilot: one paired seed (8891), 100 updates per arm, identical initialization and semantic batch schedule, 469,648-parameter causal-hybrid DADA model, free generation scoring, deterministic syntax/digit-order parser only, no solver or gold passed into inference.

The pilot produced a positive early signal: arithmetic+code IID exact 0.0% -> 1.25%, overall IID 0.56% -> 1.94%, answer-token IID NLL 1.13066 -> 1.12100, and memory IID 1.67% -> 3.33%. This is weak evidence only: absolute exact accuracy is extremely low, carry/borrow-heavy IID remained 0/62 in both arms, and counterfactual both-correct remained 0/90. Original 600-step H16 hypotheses remain open.

Real local checkpoint bytes exist for both completed pilot arms. Each has model.safetensors plus resume.pt containing optimizer and Python/Torch RNG. Reload generation matched on 64 total cases and deterministic next-update replay matched for both arms. No remote binary release was created in this run.
