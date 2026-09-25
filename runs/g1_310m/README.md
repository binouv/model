# FlyGraph G1 — 309,993,253 parameter generative bootstrap

Status at this publication: training in progress, not a completed foundation model.

This is a new causal next-token language model, initialized from scratch. It is not one of the earlier restricted-command classifiers. All 309,993,253 parameters are trainable; the output embedding is tied to the input embedding. Most parameters are in a 12-layer Transformer with grouped-query attention, RoPE, RMSNorm and SwiGLU. A 4,819,237-parameter synthetic recurrent graph workspace combines layer states at each token position. This graph is NOT a biological connectome and does not carry recurrent state across tokens. The protected-memory interface is separate and is not yet a learned tool-use capability.

## Publication scope

This directory is a separate namespace from the older R52–R61 ECC branch. Their experiment numbers and results must not be overwritten with the similarly numbered conversation experiments.

The first durable local checkpoint is step 64: 32,768 training tokens. Macro validation NLL changed from 9.344726403554281 at initialization to 4.649470395512051. This is evidence of learning the small bootstrap distribution, not evidence of general reasoning or superiority to Qwen3.5-4B. The full preregistered run is 192 optimizer steps / 98,304 tokens. An OOM killed the first attempt after its last logged step 76; steps 65–76 were not checkpointed and are excluded from the canonical trajectory. Resume starts at step 64 with activation checkpointing and unchanged batch size, sequence length and total token budget.

The local corpus contains approximately 3.74M tokens of CPython source/docstrings and synthetic RU/EN/code tasks. The external-memory API unit tests do not count as model memory accuracy. Generation is evaluated separately without gold-label input.

IMPORTANT: this source publication is NOT a claim that all historical archives or the large G1 weights have been uploaded. Binary checkpoints remain local until separately uploaded and verified by SHA256. A release manifest is not a weights release.

## Source and run status

Local environment: PyTorch 2.10.0+cpu, Python 3.13, 4 CPU cores, 4 GiB memory limit. No GPU, external training service or paid inference is used in this run. Sources and completed-checkpoint metadata are published incrementally; unfinished evaluations are never reported as completed results.
