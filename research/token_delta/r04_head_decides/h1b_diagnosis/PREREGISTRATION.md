# R04–H1b: Why did frozen Qwen3.5-4B score only 16.7%? (diagnostic follow-up)

**Registered after H1 results, 2026-10-12. Not a retroactive change to H1.**

## Prior result / scientific boundary

The external operator reports a complete R04-H1 run on an RTX 3090 using the pre-registered 1,700-instance dataset SHA256 `245cd44c8fbb6f10188079c9960c5093015622659f11a7468c166eda38916899`: 4B direct greedy **50/300 = 16.67%**, 0.8B direct greedy **1/300 = 0.33%**, H1 probe exact around 7% and validation last/MLP ~6.3% vs structured ~6.0%. These numbers are **user-reported**, not independently reconstructed here from their exact `c8e3641` local 81MB run. **H1 remains `NOT_CONFIRMED_under_registered_gates`, regardless of H1b.**

This follow-up deliberately uses **128 NEW operand-tuples**, sampled deterministically from the same arithmetic/code programmatic family but with zero overlap with all 1,700 original H1 tuples (even across language/domain), seed `20261013`; 32 examples per `(domain, language)` bucket. Thus no H1 test labels are used to tune the new experiment. H1b test prompts are still synthetic and simple; any conclusion generalizes only to this diagnostic distribution.

## Hypotheses (not decisions selected after looking at new GPU results)

- **D0 — Exact format bottleneck**: Original output scorer required *only* `p q` without surrounding words. Diagnostic arm A has a stricter score and a **final-only semantic pair score** from exactly the same raw output; if semantic accuracy exceeds strict accuracy by **at least 20 percentage points**, format errors are a major contributor to the low measured H1-style accuracy. Invalid formats are never retroactively treated as valid H1 answers.
- **D1 — Output-token truncation**: Arm A original `enable_thinking=False`, greedy, max_new_tokens=12; arm B same prompt, same decoding, max_new_tokens=96. If B's *final-only semantic* correctness exceeds A by **at least 20 percentage points**, output budget is a material problem.
- **D2 — Additional sequential computation**: Arm C `enable_thinking=True`, sample with temperature 0.6/top_p 0.95/top_k 20, max_new_tokens=384, same underlying problem. If C semantic correctness is **≥50%** and **≥25 points better than B**, extra reasoning materially helps the teacher. If it reaches ≥80%, teacher capability may support a **new explicitly multi-step latent-planning hypothesis**, **not original one-pass H1**. If C <50%, even this budget/config has not solved the task.
- **D3 — Is even the first computation difficult?**: Arm D asks for the *first intermediate digit p only*, same original tuple/operation, non-thinking greedy with max_new_tokens=32. Report D accuracy alongside A/B first-digit accuracy. This ablation changes the question and is diagnostic only, not part of H1.

### Fixed arms

| Arm | Input | Thinking | Decode | Limit |
|---|---|---|---|---:|
| `no_think_12` | Exact original question | false | greedy | 12 |
| `no_think_96` | Same original question | false | greedy | 96 |
| `thinking_384` | Same original question | true | sampled T=0.6 / p=.95 / k=20 | 384 |
| `one_step_32` | Explicit first operation only | false | greedy | 32 |

Original pinned model `Qwen/Qwen3.5-4B` revision `c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97`, Transformers 5.13.0, PyTorch 2.10.0+cu128, RTX 3090 24GB. Frozen BF16 throughout. Both 16-example pilot and full 128-case run use the same arm definitions; the pilot is engineering only and cannot be used to retune thresholds, parsing or samples. Thinking sampling uses seed `20261013 + case_index` for deterministic replay; multiple sampled seeds would be a separate ablation.

## Numeric scoring / safety against false positives

- Score the **exact full original-style numeric output** (strict, meaningful only for no-think two-digit arms) independently from a **numeric final-only** parser. Final-only looks at the end of the decoded answer or explicit `p=... q=...`; it never extracts numbers from an intermediate calculation.
- For thinking mode, the parser ONLY reads text **after `</think>`**; missing closure is marked missing, even if the unclosed thought contains two correct-looking digits. Every generated output, parse result, EOS/truncation flag, time and oracle stored as immutable per-item raw JSONL.
- Report accuracy on *all examples*, valid-response rate, first- and second-digit accuracy, RU vs EN, code vs arithmetic and operator; no filtering to model-correct responses, no test-time retries or posthoc changes.
- Do not use H1b to revise H1's original 80% teacher, 70% probe, or baseline superiority requirements.
- The new dataset is derived by code from the original H1 generator. Its canonical SHA256 is locked in `data/h1b_manifest.json` and collisions are explicitly tested.

## Research decision after diagnostic

1. **If semantic outputs are often correct but strict outputs fail:** investigate format robustness as a *new benchmark*, not silently fix the original H1.
2. **If longer non-thinking outputs help:** consider the 4B head as an autoregressive solver (extra token cost); this conflicts with the hoped-for pure one-pass plan.
3. **If only thinking helps:** develop a *new* head architecture that computes several latent reasoning states, then ask whether the 0.8B writer can render the **completed** plan. **Charge the large-model reasoning compute to inference**; do not claim one-pass or lossless equivalence.
4. **If nothing helps:** simplify/verifiably balance tasks or test a stronger head and repeat a preregistered experiment on fresh holdout cases; do not go directly to H2 bridge training using inaccurate teacher labels.

This is a falsifiable fault-isolation study, not a guarantee of positive result or GPU speedup.