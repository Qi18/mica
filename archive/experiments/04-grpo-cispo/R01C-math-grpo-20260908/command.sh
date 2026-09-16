#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES=3 python3 scripts/launch/train_phase5.py   --method grpo --base-model /data/artifacts/minimind-lab/D01-s10-preference-baseline-20260908/exported-fp32   --data /data/datasets/minimind-lab/phase5/verifiable-math-v1 --output /data/artifacts/minimind-lab/R01C-math-grpo-20260908   --run-name R01C-math-grpo-20260908 --learning-rate 3e-6 --batch-prompts 4   --num-generations 16 --max-new-tokens 8 --inner-updates 2   --beta 0.02 --epsilon 0.2 --epsilon-high 2.0 --seed 42
