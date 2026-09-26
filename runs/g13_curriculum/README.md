# FlyGraph G13 — ordered difficulty curriculum vs same-example shuffled control

Preregistered before data generation or training. G7R, G11R and G12 all left arithmetic/code near zero, so G13 changes optimization order rather than adding traces, recurrence or width.

Both arms use the same 469,648-parameter causal-hybrid model, strict final-only byte generation, optimizer recipe, 2,000 updates and **the exact same pre-sampled minibatch multiset**. The main arm receives four 500-update phases of increasing difficulty; the control receives those exact batches in a deterministic random permutation. Thus total examples and batch shapes match; only ordering relative to optimization time and learning rate changes.

Primary metric is fresh arithmetic+code IID exact. Registered gate requires at least +3 percentage points and positive direction in at least two of three seeds, with overall/transfer/regression safeguards. Fresh semantic groups must exclude G5S, G7R, G11R and all G12 groups before training. Fixed endpoints only.

No result exists yet. Confirmation would be narrow evidence that optimization curriculum matters in this small generator, not Qwen parity.
