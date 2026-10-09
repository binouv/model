# Supplemental Qwen0.8 reference, not the primary pointer hypothesis

Twenty fixed fresh IID questions: first two from each of five families x RU/EN, selected before any evaluated predictions. Exact prompts-only JSON SHA25674bfffeb6cf3014b58d14cc1402fb8e9fca28818a0962d5deeb9140e792de26c. No gold answers or executable case solver enter the inference process.

Qwen/Qwen3.5-0.8B, ggml-org BF16 GGUF SHA2569a7bed4041b7975e0f71fa34670d1e9025213bc92905ac0db75d36c4fa3fa623. llama.cpp pinned7fe450e19305b828c199d602c23a8337aaa1f03b. Native open<think> prefill;32768newtokens,40960context; recommended thinking sampling temperature1,top_p.95,top_k20,min_p0,presence_penalty1.5,repeat_penalty1. Seeds92201+i, one sample per prompt. No paid APIs, no model training in CI.

The same user instruction prefix and task strings will be evaluated on the fixed final local FlyGraph checkpoints. Native tokenizers differ and compute is not equal. FlyGraph uses greedy64tokens; actual truncation must be reported, not assumed absent. Prefix: Solve the task and return F=<integer> as the final answer.\n

Final-only parser accepts full F=integer, bare integer or boxed integer; it never searches an arbitrary last number in thought text. Qwen needs closed thinking and EOS for an uncensored result. Timeout/limit records are unknown/censored, never proof of inferiority. Exact raw generations, stopreason and timings preserved. A partial request set is NOT a completed benchmark.

This is a small synthetic reference, not general coding capability or a sufficient superiority claim. Reference outputs are evaluation only, not teacher training data. Original768-step pointer/no-copy experiment and its held criteria are unchanged.
