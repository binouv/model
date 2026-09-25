#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
python src/setup_data.py
for seed in 7301 7302 7303; do
 for variant in full shared hybrid; do
  python src/train.py --variant "$variant" --seed "$seed" --steps 2000 --batch 32 --threads 4
 done
done
for seed in 7301 7302 7303; do
 for variant in full shared hybrid; do
  python src/train.py --variant "$variant" --seed "$seed" --eval-only "checkpoints/${variant}_s${seed}/step2000.pt" --threads 4
 done
done
