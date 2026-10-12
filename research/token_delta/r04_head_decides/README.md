# R04 — Head Decides / Student Renders

**2026-10-12 status: preregistered + CPU unit-tested source only; the actual RTX 3090 GPU smoke and H1–H4 experiments have NOT yet been run here.**

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

## What to run after smoke

H1 is the first science experiment. Prepare disjoint programmatic RU/EN arithmetic and simple code-output tasks with true answers verified by a reference interpreter, run frozen Qwen3.5-4B forward and save last/mid-layer states; train linear/MLP probes against the **oracle answer** on train and evaluate heldout and hash-BOW/label-shuffle controls. Do **not** first train the 0.8B renderer to generate arbitrary text, or you risk repeating R02's failure without learning whether the head possesses a complete plan. H1 must pass before a bridge-trained student can be interpreted as receiving genuine resolved answers.

For H2–H4, use offline teacher-produced answers as *training supervision* only; never pass answers/style/length to the writer during test. Only then time a real matched-quality GPU benchmark.

## Source verification

Official [4B checkpoint](https://huggingface.co/Qwen/Qwen3.5-4B/tree/c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97), official [0.8B checkpoint](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/23eda19ab0239bc12350efba411911b8eb89f486), [Transformers Qwen3.5 API](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5), [PyTorch 2.10.0 cu128 Windows wheel](https://pytorch.org/get-started/previous-versions/).