# G6S — latent verified-trace deep supervision

Primary hypothesis: supervise verified intermediate numeric states on the prompt-end latent representation while keeping inference final-only.

Paired design: same causal d-a-d-a hybrid, same initialization within seed, same batches, optimizer, 2000 updates, byte259 vocabulary and fixed G4S/G5S semantic splits. Both arms contain the same 8-slot auxiliary head. latent_trace trains it; final_only uses auxiliary weight zero. Held data are never used for checkpoint or hyperparameter selection.

This is a small 470k-parameter mechanism probe, not the 300–800M target and not a Qwen comparison. Partial runs are never aggregated.
