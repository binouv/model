# R04 — Head Decides / Student Renders

**2026-10-12 status:** The user reports that the RTX 3090 ENGINEERING smoke passed on Windows (5/5 bridge CPU tests, 4B hidden [1,2560], writer + bridge plan [1,4,1024], peaks 7.883 / 8.074 GiB). This is **user-reported** until the original `results/smoke.json` from their local commit `72922ba` is published and its bytes checked. Scientific **H1/H1b/H1c have completed operator-reported RTX3090 runs**, with H1 negative, H1b thinking-arm censored, H1c C0/C1/C2 negative. Their exact raw checkpoints/metrics remain on local Windows commits; **H2–H4 have not been measured**.

**[Combined scientific article R02–R04 (Russian)](../SYNTHESIS_R02_R04_20261012_RU.md)** — provenance, all available results, negative evidence and original H2–H4 research plan.

## Architecture contract

- **Head (frozen):** Qwen3.5-4B, one pass over the question, decides *what* the answer means and *how detailed* it should be. No target answer/CoT in the head input.
- **Bridge (trainable):** output K continuous prefix tokens, initially K=4, mapping 2560 → K×1024.
- **Writer (frozen):** Qwen3.5-0.8B. It sees *only* the learned prefix plus a constant assistant-start marker; it is never given the original question, a length or style flag, or task-answer labels at inference.
- **Only bridge optimizer updates permitted.** The frozen writer still has knowledge and may do some computations; this experiment cannot *a priori* prove it is a purely passive printer.

The source `src/gpu_smoke.py` tests ONLY official pinned model loading, shapes, frozen params, a teacher hidden state, soft-prefix injection into the writer, and a nonzero gradient reaching the bridge through the frozen writer. Its fake engineering label is **not** the teacher's answer and yields **no scientific accuracy/speed data**.

## Contents

- `PREREGISTRATION.md` — four preregistered hypotheses with pass/fail criteria (H1 answer availability, H2 rendering, H3 head-decided verbosity, H4 end-to-end speedup).
- `MODEL_LOCK.json` — two official pinned HF model revisions, hidden sizes, hardware and packages.
- `src/latent_bridge.py` — style-independent trainable bridge + correct causal teacher-forcing loss.
- `src/gpu_smoke.py` — first engineering GPU script (BF16 RTX3090, sequential loads).
- `tests/test_bridge.py` — five CPU-only tests (no model download).
- `requirements.txt` — application dependencies; PyTorch CUDA wheel installed separately.
- `results/.gitkeep` — destination; no manufactured H1–H4 results.

## Windows PowerShell / first command

From `D:\AI\model_r04` (existing Git checkout, main):

```powershell
cd D:\AI\model_r04
git pull --ff-only
cd research\token_delta\r04_head_decides
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe src\gpu_smoke.py --lock MODEL_LOCK.json --output results\smoke.json
```

For a clean NVIDIA install, `nvidia-smi` should show the RTX 3090. The PyTorch wheel installs its compatible CUDA runtime, so **a separate CUDA Toolkit is not required** just for this smoke script. Driver 617.14 should be new enough for cu128, but the runtime preflight is authoritative. `HF_HOME` may be set to `D:\AI\hf_cache` if desired.

**Successful smoke**: `run_status=passed_smoke_no_scientific_hypothesis_tested`, head hidden `[1,2560]`, bridge `[1,4,1024]`, writer trainable params 0, bridge gradient >0, peak reserved VRAM <22 GiB for each sequential phase. **Failure**: capture full traceback and environment in a `results/` diagnostic file, do not publish `complete` status.

## H1 implementation published after the user's smoke report

**[R04/H1 one-folder starter](h1/H1_RUNBOOK.md)** — GPU pilot, frozen Qwen3.5-4B hidden layer 16/24/32 extraction, all 1,700 oracle-checked arithmetic/code questions (1,200/200/300), frozen 4B and standalone 0.8B greedy baselines, lexical/structured/shuffled-label probes, three independent readout seeds, trained-probe checkpoints and mandatory raw predictions. [Locked H1 criteria](h1/H1_PREREGISTRATION.md) retain the original R04 H1 thresholds. Eight new CPU protocol/unit tests passed on the authoring CPU; **no actual H1 GPU science was claimed**.

The local Git push from `NeoXider` returned HTTP 403 because that account lacks write access to `binouv/model`; a separate GitHub connector authorized as `binouv` published the H1 **source** directly. The original exact smoke JSON remains at local commit `72922ba` until published through authorized credentials or fork/PR.


## H1 completed negative; H1b diagnosis (2026-10-12)

The RTX 3090 operator reports a fully completed frozen-head H1: **Qwen3.5-4B 50/300 (16.7%)** on strict two-digit greedy questions against the preregistered **80%** head gate; the best probes also failed the >=70% accuracy and baseline-superiority gates. **H1 is NOT_CONFIRMED**, not merely incomplete. This outcome is [documented separately](h1/H1_RESULT_REPORTED_20261012.md) as an **external user-reported result** pending synchronization of their exact 33 checkpoints and full raw predictions from local commit `c8e3641`. Do not rewrite the original [H1 preregistration](h1/H1_PREREGISTRATION.md).

[H1b — preregistered fresh-data diagnosis](h1b_diagnosis/PREREGISTRATION.md) runs **128 new examples disjoint from all original H1 operands** and tests strict-format failures, 12 vs 96 token caps, 384-token thinking mode, and an easier one-step control. [Execution guide](h1b_diagnosis/README.md). The H1b CPU tests passed (14/14). The operator subsequently reported an **RTX 3090 diagnostic pilot (16 cases)**: no-thinking 12 tokens 3/16, no-thinking 96 tokens 3/16, thinking 384 tokens 0/16, and one-step control 10/16; original H1 outputs had zero invalid-format cases. The **full 128-case H1b run completed locally**: no_think_12 and no_think_96 both 21/128, one_step_32 67/128; thinking_384 had zero closed responses and 100% truncation, so reasoning accuracy was **not evaluable**. See [operator-reported full H1b](h1b_diagnosis/FULL128_USER_REPORTED_20261012.md). Local commit `33f20ab` is not synchronized. H2 remains untested, and must not be presented as verified until the head can supply correct plans at a competitive quality threshold.

## H1c — trainable latent computation, source complete (2026-10-12)

[R04-H1c complete GPU runner](h1c_latent_compute/README.md): an independently locked **1,700-case** 1200/200/300 dataset, frozen Qwen3.5-4B pre-answer features, recurrence depths 0/2/8, lexical and structured controls, label-shuffle negative control, a confidence-gated **real early exit**, and all trained-probe/checkpoint publication logic. [Original preregistration](h1c_latent_compute/PREREGISTRATION.md). **13/13 CPU unit tests passed**, including confirmation that skipped GRU steps actually are not computed and early outputs match the full trajectory. The CPU workflow verifies the published dataset bytes. **H1c completed locally on RTX3090; C0/C1/C2 are all False**: Qwen T=0 around 8–9% exact, T=8 around 6–8%; 1700-case feature extraction confirmed. See [operator-reported H1c outcome](h1c_latent_compute/H1C_USER_REPORTED_RESULT_20261012.md). Local commit `1e4696c` remains unpublished; original raw files are still needed to verify all seeds and checkpoints. H1 remains negative and is not re-opened by H1c.

## After H1/H1b/H1c: test original H2–H4 without changing gates

**[H2 executable GPU source + runbook](h2_bridge_transfer/README.md)** — 12/12 CPU unit tests, new disjoint dataset and SHA locked, staged/pre-registered head→writer transfer with no question/style leakage. [Original H2 scientific preregistration](h2_bridge_transfer/PREREGISTRATION.md) retained unchanged. — frozen Qwen3.5-4B/0.8B, no question or style leaked to writer, fresh disjoint dataset, original H2 gates unchanged. Oracle-plan is an engineering upper-bound and no H2 GPU scores exist.  
**[Transfer original local commits / raw checkpoints](LOCAL_ARTIFACT_IMPORT_GUIDE.md)** — NeoXider fork branch publication instructions and exact missing H1/H1b/H1c artifacts.



H1 did **not** pass its registered gates. H2 is still useful as a transmission/negative-control experiment on a fresh split, but faithful imitation of a teacher with 16.7% accuracy cannot meet the original 75% all-test quality gate on the same task distribution. Never train on the ~50 teacher-correct **test** examples; evaluate teacher-correct as a conditional **reporting subset**, not a training set. H2a may separately use explicit oracle plans as an upper-bound engineering control, not as scientific evidence that the head computed the plan.

For H2–H4, use offline teacher-produced answers as *training supervision* only; never pass answers/style/length to the writer during test. Only then time a real matched-quality GPU benchmark.

## Source verification

Official [4B checkpoint](https://huggingface.co/Qwen/Qwen3.5-4B/tree/c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97), official [0.8B checkpoint](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/23eda19ab0239bc12350efba411911b8eb89f486), [Transformers Qwen3.5 API](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5), [PyTorch 2.10.0 cu128 Windows wheel](https://pytorch.org/get-started/previous-versions/).