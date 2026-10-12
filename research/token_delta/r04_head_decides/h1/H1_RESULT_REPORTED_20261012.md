# R04/H1 — Completed negative experiment (external RTX 3090 operator report)

**Outcome:** `NOT_CONFIRMED_under_registered_gates`. **Do not change the original predeclared thresholds.**

This report records **measurements supplied by the external RTX 3090 operator on 2026-10-12**, not a fresh GPU inference run by the publishing agent. Original generated results/33 checkpoints were reported committed locally as `c8e3641` (~81MB), but their *exact binary bytes have not been independently retrieved/published through this connected GitHub session*. Therefore this document is **not an independent verification receipt** or a substitute for the real `h1_metrics.json`, original predictions, checkpoint shards, manifest and SHA256.

## Reported environment / dataset

- RTX 3090 24GB; Torch CUDA 2.10.0+cu128; Transformers 5.13.0; pinned original Qwen3.5-4B and Qwen3.5-0.8B.
- Data SHA256 canonical `245cd44c8fbb6f10188079c9960c5093015622659f11a7468c166eda38916899`; 1700 samples (1200/200/300), from the repository-verified H1 code.
- Engineering pilot 8 cases passed, max reserved 8.83GiB.
- Frozen layers `[16,24,32]` BF16 `[1700,2560]`, aggregate cache SHA reported starting `482e2d20` (full 64-digit SHA is unavailable in this report; **do not treat prefix as a complete hash**).
- Direct greedy teacher baseline **50/300 = 16.67%**, writer-only baseline **1/300 = 0.33%**. Required teacher gate **80%** -> **failed**.
- Probe training 600 updates, three seeds; best validation `last:mlp` about **6.3%** exact vs structured control **6.0%** and shuffled-label control **1.8%**. On test selected seeds reportedly **22/300, 21/300, 17/300** (~7.33%, 7.00%, 5.67%); user also reported `digit0~55%`, `digit1~12%` (exact per-seed data pending). Required probe threshold >=70% and superiority gates -> **failed**.
- All 33 probe/training-state checkpoints and metrics reportedly saved in local commit `c8e3641`, push blocked with HTTP 403 because Windows git identifies as `NeoXider` (not the writable connected `binouv` account).

## Scientific conclusion

Original H1 hypothesis, namely that frozen Qwen3.5-4B *at the last pre-answer prompt token* contains an already-computed **two-step answer** decodable by simple trained readout, **is NOT CONFIRMED under predeclared gates**. Lower threshold, changed parser or different thinking mode must never be silently used to claim H1 passed.

The low teacher-only accuracy needs fault isolation before H2. Plausible explanations include exact-format failures, `max_new_tokens=12`, disabled thinking and the computational difficulty of two sequential modulo operations. The new follow-up [H1b](../h1b_diagnosis/PREREGISTRATION.md) uses **128 fresh operand tuples disjoint from all 1700 H1 cases** and does not rewrite this result. We cannot tell from the aggregate figures alone which failure mechanism dominates; raw head predictions and valid/invalid response rates are necessary.

## Exact artifact publication instructions

Keep local commit `c8e3641` reachable (`git branch r04-h1-full c8e3641`), do not force-push over main. For a standalone fork, push your branch to `NeoXider/model` then create a PR into `binouv/model`. Alternatively authenticate Git locally as `binouv` (if your account) and push a unique branch. After upload, independently fetch the raw test prediction/receipt files, verify full SHA256 and check all 33 actual checkpoint files against the manifest before declaring publication verified.