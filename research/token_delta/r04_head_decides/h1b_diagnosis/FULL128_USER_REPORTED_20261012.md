# H1b full128 — user-reported completed GPU diagnostic (2026-10-12)

**Publication integrity status:** User reported the complete RTX 3090 results in chat; original raw `diagnosis_predictions.jsonl`, `diagnosis_metrics.json` and local Git commit **`33f20ab` have NOT been transferred to this GitHub repository or independently replayed here**. This is a provisional run receipt, not a claim of verified binary/raw publication. The local `NeoXider` credentials lack write permissions to `binouv/model` (HTTP 403).

## Source and method

- Pre-registered H1b diagnostic: [PREREGISTRATION.md](PREREGISTRATION.md), [GPU runner](src/h1b_run.py).
- Frozen, pinned `Qwen/Qwen3.5-4B`, RTX 3090, Transformers 5.13.0, BF16.
- **128 fresh H1b cases** disjoint from the original 1,700 H1 problems; H1b seed=20261013. This is NOT a reopened H1 test.
- Analysis of the original 300 H1 outputs: **invalid_format = 0**, so strict formatting was not the cause of the low exact score.
- Status of the original R04/H1: **`NOT_CONFIRMED_under_registered_gates`** — unchanged.

## Full128 results as reported by operator

| Registered diagnostic arm | Correct | Fraction | Completion / interpretation |
|---|---:|---:|---|
| `no_think_12` | 21/128 | 16.40625% | Model typically stops after ~4 generated tokens |
| `no_think_96` | 21/128 | 16.40625% | Same score and stopping behavior: extra budget unused |
| `thinking_384` | 0/128 | 0% scored | **0% parseable, 100% truncated at exactly 384 tokens, no `</think>` closure** |
| `one_step_32` | 67/128 | 52.34375% | Explicit single-step question, not the same task as two-step |

Operator-reported stratification: arithmetic ~30% exact vs Python-code output ~3% in the full diagnostic; precise subgroup denominators and arm identity must be reconciled against raw files before formal table publication.

## Gate-level conclusions

- **D0 (strict format was the bottleneck): rejected.** Original 300 raw responses showed zero invalid-format cases.
- **D1 (12-token truncation impaired non-thinking performance): rejected.** 12→96 tokens changes neither correctness (21/128) nor actual output length; early model EOS/stopping, not output cap.
- **D2 (additional serial thinking improves answer correctness): INCONCLUSIVE / NOT EVALUABLE.** The 384-token thinking arm never emitted `</think>` and never reached final-answer generation. Its reported 0% is **completion at 384 tokens**, NOT proof of zero reasoning capability.
- **D3 (first computation is easier): supported descriptively** (67/128 one-step vs 21/128 two-step in different task instructions), but is not causal proof that the second digit was already computed during prefill.

## Next action order

1. Preserve original local `33f20ab` and its actual raw artifacts. Publish their exact bytes via authorized `binouv` account or a `NeoXider` fork/PR; do not invent missing hashes.
2. R04-H1c frozen Qwen latent-recurrence GPU run is the **primary next scientific test**, using its separate locked dataset and 0/2/8-step recurrence controls. [H1c source](../h1c_latent_compute/README.md).
3. Optional independent **H1b-D2r** thinking-budget diagnostics later: 1024/2048 generated tokens, close-marker and EOS accounting, with a new fresh sample or an explicitly labelled exploratory re-evaluation. Do not alter H1b predefined D2 gate after seeing these results.

All items above are based on reported measurements and on audited published source; this file is **not** a substitute for the original per-example output, software environment, and checkpoint verification.
