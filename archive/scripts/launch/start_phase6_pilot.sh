#!/usr/bin/env bash
set -euo pipefail
cd /data/projects/minimind-lab
BASE=/data/artifacts/minimind-lab/D01-s10-preference-baseline-20260908/exported-fp32
PY=/data/venvs/minimind-eval/bin/python
DATA=/data/datasets/minimind-lab/phase6/record-agent-v1
RUNS=/data/artifacts/minimind-lab/phase6-launch-20260908
mkdir "$RUNS"
# Preserve all resident processes; use only our newly launched PID handles.
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 nohup "$PY" -u scripts/eval/eval_phase6_agent.py \
  --model "$BASE" --tasks "$DATA/val.jsonl" \
  --output /data/artifacts/minimind-lab/A00-s10-agent-baseline-20260908/validation-agent \
  --run-name A00-S10-Agent-Baseline-Val-20260908 > "$RUNS/a00.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUNS/a00.pid"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 nohup "$PY" -u scripts/launch/train_phase6_sft.py \
  --config configs/agentic_rl/phase6-a01-pilot.json \
  --output /data/artifacts/minimind-lab/A01-agent-sft-pilot-20260908 \
  --run-name A01-Agent-SFT-Pilot-20260908 > "$RUNS/a01.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUNS/a01.pid"
printf 'A00 pid=%s; A01 pid=%s\n' "$( < "$RUNS/a00.pid")" "$( < "$RUNS/a01.pid")"
