# G3: preregistered multi-architecture cognition screen

One finite campaign: full4-block causal attention; shared2-block attention applied twice; hybrid2gated-delta+2attention. Trainable parameters1,901,696 /1,475,200 /1,903,760. These are SMALL GENERATIVE PROBES initialized from scratch, NOT relabelled310Mweights and NOT broad foundation models. G2R310Mweights remain unchanged.

Each architecture is trained2000updates,batch32,three seeds7301/7302/7303 with the same ordered training examples/data-only sampler73311. AdamWlr.001,warmup100,cosine-to15%,beta(.9,.95),wd.01. Train16384,validation400; four fixed G2held suites160/80/80/80. Stronger canonical grouping excludes commuted arithmetic/list/per-key memory equivalents from train versus held. Exact tokenizer is reused from G1; no model weights imported. Inputs share `Return the final integer.\n`. Answer-only masked-position CE still uses the full8192vocabulary. No solver at inference.

H1: shared iterative weights improve IID by>=5pp over full, positive surface/extrapolation gain,no>2ppcounterfactual-pair regression,direction in>=2/3seeds. H2:same criterion for persistent delta hybrid; reset-state inference is its single mechanistic control. Four executed blocks in all models; actual FLOPs/wall time differ and must be reported. Fixed step2000finals only, no best-test model picking. These criteria do NOT establish Qwen parity or general intelligence.

The attached one-shot public GitHub Actions benchmark evaluates the same old40fixedsyntheticquestions with Qwen3.5-4B Q4_K_M in ACTUAL THINKING mode. Official native `<think>\n`prefill, no empty closed-think prefill; recommended general-task sampling;32768newtoken limit and40960context. Every truncation/missing thought closure is recorded; truncated runs cannot establish superiority. Native tokenizers and inference compute differ. No27Bbaseline. No paid API, tools or generated-code execution.

Training is currently local and in progress. Sources/results will be published after completed checkpoints. A protocol/manifest is not a weight upload. Do not confuse G3 with existing R-series numbering.
