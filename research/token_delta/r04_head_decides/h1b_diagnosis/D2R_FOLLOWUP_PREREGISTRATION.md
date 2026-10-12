# H1b-D2r — Thinking-budget survival / completed-answer diagnosis (independent follow-up)

Registered **after** the user-reported H1b full128 run on 2026-10-12. **This is not the original H1 or H1b D2 result**, and it must not retroactively alter their preregistered thresholds or scores. This note is a *protocol only*: no H1b-D2r GPU results and no D2r runner are claimed yet.

## Observed censoring to diagnose

H1b full128 on frozen Qwen3.5-4B reported 0/128 final answers at 384 generated tokens with `enable_thinking=True`; all 128 samples hit the generation cap and zero samples closed `</think>`. This is **100% right-censoring of answer generation**, not empirical zero reasoning accuracy. Non-thinking 12 and 96 caps gave the same 21/128 with early EOS, excluding truncation in those two arms. First-step-only result 67/128.

## Primary scientific question

If budget is enlarged while keeping the pretrained official model, prompt wording, seed schedule, and sampling otherwise fixed, does thinking mode **ever** close and produce a valid final answer? If so, is the completed answer correct on separately heldout two-step questions?

## Predeclared independent protocol

- Freeze `Qwen/Qwen3.5-4B` at model revision `c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97`. Same BF16 RTX3090 environment.
- Use an independently regenerated *fresh set of operand tuples with no exact group overlap with H1 (1700), H1b (128), or H1c (1700)*. Group code and seeds must be locked and hashed before running. Keep each (domain, language) stratum represented. Do not reuse the 128 seen H1b examples for a confirmatory claim.
- Pilot 16 independent tasks (4 per arithmetic/code × RU/EN stratum), followed by a disjoint 64-task test (16 per stratum). Pilot measures mechanics only and is **never** used for threshold changes.
- Budget arms, identical prompts: 1024 and 2048 `max_new_tokens`, `enable_thinking=True`, `do_sample=True`, `temperature=.6`, `top_p=.95`, `top_k=20`. Exactly preserve the original H1b thinking sampling to isolate the budget change. An optional 4096-token third arm is a **separately labeled exploratory** configuration; do not silently add its results to the primary 1024/2048 comparison.
- Official Qwen3.5 card separately recommends general-task thinking `temperature=1.0, top_p=.95, top_k=20` with a presence penalty, and .6 for precise coding. This is a **different sampling-configuration ablation** and must not be conflated with budget changes. Transformers API does not automatically provide the same presence-penalty implementation as serving APIs.
- Decode tokens using `skip_special_tokens=False`; count the exact `</think>` close-marker ID and its earliest occurrence. Verify a non-empty post-close final response. Also record eos/im_end, output token count, cap hit, and raw answer text (or a hash-preserved complete record).
- Score two independent measures on **all** test questions: (a) closure + final-answer valid-output rate; (b) semantic exact of complete final answers, with all unclosed runs counted uncompleted, not numerically wrong as reasoning. Separately show conditional accuracy among completed answers, with numerator and denominator. Strict original H1 scores cannot be recomputed.
- Minimum **80% closure** on test is required before calling final-answer correctness an adequately measurable property under that budget; otherwise D2r answer accuracy remains **INCONCLUSIVE: budget-censored**. A conditional >50% on 2 examples is not sufficient. Analyze time, tokens-per-final, and RTX3090 VRAM peak as first-class costs.
- Reports include 16/64 pilot/test separation, complete per-example predictions, all failed traces, fixed random seeds, original revision, parser version, SHA256 and exact CUDA/PyTorch/transformers versions.
- If 2048 still yields few completed answers, move to a new preregistered **forced-exit / two-pass thinking-budget** experiment. Do not fabricate a closing marker post hoc and claim the model naturally ended thinking. A forced exit costs additional generation and yields a different model-output distribution.

## Model-card rationale and limitations

Official Qwen3.5-4B: https://huggingface.co/Qwen/Qwen3.5-4B
The model card recommends significant total generation budgets for reasoning, and distinguishes thinking/nonthinking sampling parameters. A larger `max_new_tokens` cap only allows more completion; it is not equivalent to guaranteeing a fixed thinking budget or a natural `</think>` closure.

## Scientific order

**Run the already published H1c GPU experiment first**: it uses a separate locked dataset and directly tests 0/2/8 auxiliary computations plus lexical/structured controls. H1b-D2r is a follow-up after H1c, on disjoint fresh tasks. Neither can change the original negative H1 conclusion.
