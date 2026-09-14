#!/usr/bin/env bash
set -euo pipefail
cd /data/projects/minimind-lab
export CUDA_VISIBLE_DEVICES=0
PY=/data/venvs/minimind-eval/bin/python
OUT=experiments/08-inference/I01-I04-a10-20260914
# Original fixed-threshold BF16 failure retained; main run is fixed reference-trace replay.
"$PY" scripts/eval/benchmark_kvcache_gqa.py --out "$OUT" --mode full
