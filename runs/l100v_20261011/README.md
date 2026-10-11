# L100V — verified-skill continuation of an actual100M FlyGraph

Local-only full-parameter training from recovered EC BF16 export A. New optimizer: **not exact EC/CS resume**. One fixed combined recipe; no causal single-factor claim and no Qwen superiority claim.

## Reproduce

Use Python3.11+ with torch2.10.0, safetensors0.7.0, pytest. Put the verified parent model and config in a local `parent` folder. Reuse the original8192-token tokenizer.

```bash
python src/data.py --ancestor-data /path/to/L100M/data
pytest -q tests
python src/evaluate.py --model parent/model.safetensors --config parent/config.json --name parent
python src/train.py --parent parent/model.safetensors --parent-config parent/config.json
python src/evaluate.py --model checkpoints/step_2048 --name final
```

Data builder refuses to overwrite existing splits. Resume a durable run checkpoint with `--resume checkpoints/step_NNNN`; do not rerun completed training. The code evaluator accepts only bounded straight-line affine functions; it is not a general coding benchmark. No oracle is used by the model generation interface.
