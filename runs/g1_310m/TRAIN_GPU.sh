#!/usr/bin/env bash
# GPU continuation recipe. Created here, NOT executed on a GPU in this run.
# No network download and no automatic spending. Supply a curated local corpus first.
set -euo pipefail
cd "$(dirname "$0")"
: "${CHECKPOINT:?Set CHECKPOINT to a complete FP32 checkpoint directory}"
: "${FINAL_STEP:?Set an explicit total optimizer step budget}"
export OMP_NUM_THREADS=4
python src/train.py --resume "$CHECKPOINT" --device cuda --amp bf16 \
  --steps "$FINAL_STEP" --seq 1024 --batch 1 --threads 4 --lr .01 \
  --eval-every 250 --save-every 250
