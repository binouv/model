# L100GE — real generative 100M readout LoRA continuation

Baseline is physically verified L100EC curriculum BF16 on GitHub, 100,028,328 parameters. The 12,288-parameter FP32 LoRA is attached to the LM **final hidden state before tied vocabulary projection**. It modifies actual autoregressive generation; no parser/solver appears in the inference path.

This is an **additional trainable adapter**, not full dense fine-tuning. Both baseline and adapted versions use the exact same frozen 100M base; data sources are immutable L100A synthetic replay/memory with RU/EN and code. Qwen3.5-0.8B general superiority is **not established**.

## Checkout replay

From repository root:

```bash
cd runs/l100ge_20261012
tar -xJf artifacts/source_data_results.tar.xz
python src/assemble_parent.py
python -m pip install torch safetensors pytest
pytest -q tests
python src/run.py --eval
```

For an extracted archive outside the repo, first reconstruct `parent_runtime/models/curriculum/model.safetensors` with the checked-in `runs/artifact_recovery_20261012/checkpoint_parts.py restore` command from the repository, or set `FLYGRAPH_PARENT_RUNTIME` to an already verified L100EC runtime directory. `src/run_executed.py` is the exact training-time source with container-specific paths; `src/run.py` changes only paths to packaged equivalents (training math identical). Training corpus files are real bytes, not seed-only promises.

Native resume checkpoints every64 steps contain `adapter.safetensors`, `training_state.pt` with optimizer/RNG/sampler, and a SHA manifest. `--train` always starts from a new run, so do **not** rerun a completed training stage in place. Restoring from a checkpoint must use the stored RNG/optimizer state, not a fresh warm-start. Results are published in `reports/REPORT_RU.md` and raw `metrics/PREDICTIONS.jsonl`.

### Direct generation (no inference solver)

```bash
python src/infer.py --arm readout_lora --prompt "What is 2 + 3?\nAnswer:" --max-new-tokens 24
python src/infer.py --arm parent --prompt "What is 2 + 3?\nAnswer:" --max-new-tokens 24
```

**Scientific result:** negative. On preregistered 160 held prompts, parent 3/160 strict exact and adapter 1/160; no demonstrated generalization improvement. Test code includes real checkpoint optimizer/sampler/RNG replay and prompt-only generation. `artifacts/all_native_checkpoints.tar.xz` carries native optimizer states at 64/128/192/256 with manifest hashes; large frozen parent must be reconstructed from the *already published* verified original 100M GitHub parts, not substituted with a metadata hash.