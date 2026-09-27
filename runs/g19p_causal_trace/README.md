# G19P causal trace-order pilot

This branch starts from completed G18P commit `4cd2b15`.

The registered comparison is **noop trace vs same-multiset numerically sorted bag trace vs chronological ordered trace**. All arms use the same 491,072-parameter scratch recurrent language model, the same fixed training schedules within seed, and exactly four generated trace slots before the final answer.

Generation receives only the prompt. Gold traces and answers are used only for training targets and post-generation scoring. The held splits do not select checkpoints.

A prior 5-step local smoke existed only to validate execution mechanics. It is excluded from all research claims.

This is a small synthetic mechanism pilot, not the 300–800M target and not evidence of Qwen parity.
