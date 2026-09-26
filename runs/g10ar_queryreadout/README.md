# G10AR — parameter-matched query-readout architecture screen

Preregistered before training. This is a **new namespace** because earlier `G10A` only stored a preregistration and its exact executable/data state was never run. G10AR fixes an explicit memory address space `w/x/y/z`, structurally disjoint from the historical `a/b/c/d` memory groups.

Both arms start from the actual released G8C `verified_trace` step-2000 parent for seeds 7601/7602/7603. The 469,648-parameter backbone is frozen. Only a 36,865-parameter adapter is trainable; total parameters are 506,513 in both arms.

The main arm adds causal cross-token readout attention after the G8C final norm. The control adds a parameter-matched per-token MLP at the same location. Both use identical positive NLL, examples, update count and sampler. The comparison therefore tests cross-token query-conditioned readout rather than extra parameter count or a contrastive loss.

Fixed 200-update endpoints; no held-based selection. Real checkpoints must include combined model bytes plus adapter optimizer/RNG. The experiment is a narrow mechanism screen, not evidence of Qwen parity or a 300–800M result.
