# L100P: local 100M learned prompt-copy experiment

Status: registered, local pointer-arm training started; NOT a completed result.

Parent: verified local L100M step1024 BF16 bytes, SHA256 b2112e21dcc653780da5c6f36137c0deab3f2fcc01bb04662eb518cf4a07b3ec. Upcast to FP32 with NEW Adafactor because the original FP32/optimizer archive is not mounted. This is not exact ancestor resume.

One distinct architectural hypothesis: learned query/key attention over causal prompt hidden states provides a source-token copy distribution mixed with unrestricted vocabulary probabilities, improving independently generated memory/list answers and counterfactual pairs. There is no key-search parser, arithmetic solver or gold answer in inference.

100,127,401 stored parameters per arm, 99,073 new copy parameters. Main has all100,127,401 functionally active; no-copy control has100,028,328 active backbone parameters and computes the same copy graph with gate multiplied by zero. Its copy-only weights have zero gradient. Not a claim that inactive control parameters learned.

Both arms start from exactly the same available ancestor, use the same16,384 preserved training cases, targets F=<integer> plus EOS,768 fixed optimizer updates,batch4 and identical sampler/LR/shapes. Fresh held:400IID,200surface,200extrapolation,200memory-counterfactual members/100pairs; original120probes separate regression only. Canonical groups are disjoint from every available L100M case. No natural-language pretraining corpus or Qwen model is executed in this small continuation stage.

L100C is a separate contrastive experiment whose main has already completed and whose control is marked running in Git. Its work is not duplicated or overwritten by L100P. The architecture and objective differences are registered separately.

Twelve local engineering tests passed before training. Goal remains own100–300M model better than Qwen3.5-0.8B; no such result currently established. Model checkpoints/data/source archives will be saved locally with hashes. Git status/manifest is not actual weight-byte upload.

Concept reference: See et al., ACL2017, https://aclanthology.org/P17-1099/. Original implementation here; no third-party source files or pretrained weights copied.
