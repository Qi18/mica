#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES=5 python3 scripts/eval/eval_phase5_math.py   --model /data/artifacts/minimind-lab/D01-s10-preference-baseline-20260908/exported-fp32 --data /data/datasets/minimind-lab/phase5/verifiable-math-v1 --split test   --output /data/artifacts/minimind-lab/R01A-s10-math-baseline-20260908/eval/test   --candidate R01A --sample-k 4 --seed 42
