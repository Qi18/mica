#!/usr/bin/env bash
set -euo pipefail
cd /data/projects/minimind-lab
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
EXP=experiments/08-inference/I01-I04-a10-20260914
DEST=${1:?Usage: bash reproduce.sh NEW_OUTPUT_DIRECTORY}
test ! -e "$DEST" || { echo "Use a new output directory" >&2; exit 1; }
mkdir -p "$DEST"
# Preserve the original numerical qualification and its failure; do not weaken thresholds.
cp "$EXP/correctness.json" "$EXP/precision-diagnosis.json" "$EXP/initial-config.json" "$DEST/"
/data/venvs/minimind-eval/bin/python scripts/eval/benchmark_kvcache_gqa.py --out "$DEST" --mode full
