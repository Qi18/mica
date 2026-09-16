#!/usr/bin/env bash
set -Eeuo pipefail
cd /data/projects/minimind-lab
gpu=2
used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
test "$used" -lt 1000 || { echo "Target GPU busy" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$gpu"
exec flock -n /tmp/minimind-phase4-gpu"$gpu".lock /data/venvs/minimind-eval/bin/python scripts/launch/train_phase4.py \
  --method sft --data /data/datasets/minimind-lab/phase4/official-dpo-v1 \
  --base /data/artifacts/minimind-lab/S10-ifeval-curriculum-v4-20260907/checkpoints/s08_best_val_768.pth \
  --output /data/artifacts/minimind-lab/D02-chosen-only-sft-20260908 \
  --run-name D02-Phase4-sft-S10 --lr 4e-8 --beta 0.15 --batch-size 8 --global-batch 32
