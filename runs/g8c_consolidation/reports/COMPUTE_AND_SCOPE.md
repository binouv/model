# Compute and scope

G8C has 469,648 total/active trainable parameters per model. It is not new 300–800M training. Six continuations preserve three ancestral learned states with AdamW, torch, Python and sampler RNG. The new objective and learning-rate schedule are explicitly registered.

Both continuation arms receive identical family/prompt-length batches, tensor shapes, supervised positions and 1,000 optimizer updates in stage two. This is not an exact hardware FLOP measurement: full triangular-solve backward and attention-kernel FLOPs were not instrumented. Actual step wall time, input/padded/supervised token counts, generation latency and output byte counts are saved.

The original trace training used more tokens and computation. Consequently the entire two-stage comparison is NOT compute-matched. Its benefit cannot be attributed solely to supervision format at equal complete training cost.

The model receives only prompt strings; cases and answers belong to the post-generation evaluator. Both arms may emit an optional trace followed by one terminal F=integer, with EOS required. Exact internal reasoning trace is not a success metric after final-only consolidation. Correct answers alone do not prove correct hidden reasoning.

1,600 questions include 400 dependent counterfactual members (200 pairs). Questions repeat across the three ancestral seeds. Do not describe them as 4,800 independent held examples per arm. New groups are disjoint from available G5S cases, not every inaccessible earlier corpus.

The completed Qwen3.5-4B thinking run used a different forty-question fixture. Its percentage cannot be compared directly to the 1,600-case G8C suite, and its quantization is not a BF16 ceiling. No new teacher or Qwen model was run in G8C.
