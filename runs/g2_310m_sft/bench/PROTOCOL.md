# Completed external Qwen baseline: limited and explicit

Actual public GitHub Actions run36186603496/job108241335102 completed all40predictions. Exact raw strings, EOS status, model SHA256, runner implementation commit and fixture hash are inQWEN_RESULT.json. Full logs and the exact inference code are preserved by commit450d46f7cfaa0e8ee6954ab190abeb14e7f69c1c on branchflygraph/g2-qwen-smoke. No paid API, model tools, grammar restriction or generated-code execution.

Qwen3.5-4B Q4_K_M, no-thinking,64newtokens, greedy. ALL compared FlyGraph checkpoints receive the SAME prefix: `Give only the final integer. Do not explain.\n`, same forty examples and64token cap. Each uses its native tokenizer/input convention. This is not equal FLOPs, full-precision Qwen, or a model capability ceiling. The40case subset was fixed before G2 held results (first8IIDcases per family).

The previous8token smoke run completed but all40answers were truncated while producing explanations. It is quarantined and NOT interpreted as zero intelligence. The subsequent shared-format64token check is labelled supplemental/post-smoke and does not replace any preregistered G2 primary test. Qwen outputs are EVALUATION ONLY and never used as training data.

Result: Qwen32/40, completed G2answer-only arm1/40. The latter's400case primary evaluation is separately saved. No superiority or parity claim is justified.
