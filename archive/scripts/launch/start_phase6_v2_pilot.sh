#!/usr/bin/env bash
set -euo pipefail
cd /data/projects/minimind-lab
PY=/data/venvs/minimind-eval/bin/python
RUNS=/data/artifacts/minimind-lab/phase6-v2-launch-20260908
mkdir "$RUNS"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 nohup "$PY" -u scripts/eval/eval_phase6_agent.py \
 --model /data/artifacts/minimind-lab/A01-agent-sft-pilot-20260908/checkpoint-300 \
 --tasks /data/datasets/minimind-lab/phase6/record-agent-v2/val.jsonl \
 --environment phase6_agent_v2 \
 --output /data/artifacts/minimind-lab/A01-v2-transfer-baseline-20260908/validation-agent \
 --run-name A01-v2-Transfer-Baseline-Val-20260908 > "$RUNS/baseline.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUNS/baseline.pid"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 nohup "$PY" -u scripts/launch/train_phase6_sft.py \
 --config configs/agentic_rl/phase6-a01-v2-pilot.json \
 --output /data/artifacts/minimind-lab/A01-agent-sft-v2-pilot-20260908 \
 --run-name A01-Agent-SFT-v2-Pilot-20260908 > "$RUNS/sft.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUNS/sft.pid"
printf 'baseline pid=%s; sft pid=%s\n' "$( < "$RUNS/baseline.pid")" "$( < "$RUNS/sft.pid")"
