#!/usr/bin/env bash
set -Eeuo pipefail
cd /data/projects/minimind-lab
python3 scripts/launch/train_phase5.py --method cispo --reward-type integer --base-model "$R03_WARMSTART" --data /data/datasets/minimind-lab/phase5/verifiable-mod10-v1 --output /data/artifacts/minimind-lab/R03D-mod10-cispo-20260908 --run-name R03D-mod10-cispo-20260908 --learning-rate 5e-7 --batch-prompts 10 --num-generations 8 --max-new-tokens 4 --rollout-temperature 1.0 --rollout-top-p 1.0 --inner-updates 2 --beta 0.05 --entropy-coef 0.002 --stratify-by-answer
