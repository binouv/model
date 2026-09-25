# G2R-310M: durable-weight recovery and paired SFT

Status: RUNNING, not a completed model result.

This run starts from the physically mounted G1 step192 BF16 weights, verified by the saved shard hashes, then upcasts to FP32 and initializes a NEW Adafactor optimizer. It does not claim to resume G2 step64: that checkpoint's metadata is in Git, but its weight bytes are not mounted or published. The separate g2_310m_sft directory is preserved, not overwritten.

One hypothesis: completion-only verified-answer loss improves unrestricted exact-answer generation versus full-sequence loss on the same examples, starting weights and update count. Two trained arms, each 256 updates of batch4, all309993253parameters trainable, no architecture changes. Baseline is the same G1 export upcast to FP32. New optimizer, seed7201, sampler72011, learning rate .006, 16step warmup and cosine decay to10%, gradient clipping1, checkpoints every64steps.

The six G2 dataset JSONL files were regenerated locally and match ALL published SHA256 values from commit37b2fdebfc664720d75131646ce9267dafc4d79a. Train4096, validation160, testIID160, surface80, extrapolation80, counterfactual80/40pairs. Semantic groups remain split. No external teacher or paid API was used.

Success: IID exact gain>=5pp versus both G1 and full-sequence control, positive combined surface/extrapolation gain and positive both-correct counterfactual-pair gain, with a paired exact sign-test p<.05 for statistical confirmation. Evaluation uses only prompts, full8192 vocabulary, greedy generation,8newtokens and no tools. No interim test checkpoint selection.

Publication audit: GitHub Releases returned an empty list. G1 tokenizer bytes ARE now in runs/g2_310m_sft/data/tokenizer_xz, superseding the older G1 publication-status snapshot. Large G1/G2 weight files and all C1 archives are NOT all uploaded. This source/protocol commit is not a weight release. Container git clone failed DNS; the authorized GitHub connector remains functional for Git-object writes.

Historical raw96 R52-R56 and repository ECC R52-R61 are different lineages. Their run numbers are not merged or used as numerical baselines for this language-model experiment.
