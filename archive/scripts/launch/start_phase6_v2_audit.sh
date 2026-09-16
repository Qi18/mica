#!/usr/bin/env bash
set -euo pipefail
cd /data/projects/minimind-lab
PY=/data/venvs/minimind-eval/bin/python
MODEL=/data/artifacts/minimind-lab/A01-agent-sft-v2-pilot-20260908/checkpoint-225
DATA=/data/datasets/minimind-lab/phase6/record-agent-v2
LOG=/data/artifacts/minimind-lab/phase6-v2-launch-20260908
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 nohup "$PY" -u scripts/eval/eval_phase6_agent.py \
 --model "$MODEL" --tasks "$DATA/val-counterfactual.jsonl" --environment phase6_agent_v2 \
 --output /data/artifacts/minimind-lab/A01-v2-counterfactual-20260908 \
 --run-name A01-v2-Counterfactual-Val-20260908 > "$LOG/counterfactual.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$LOG/counterfactual.pid"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 nohup "$PY" -u scripts/eval/probe_phase6_rollouts.py \
 --model "$MODEL" --data "$DATA" \
 --output /data/artifacts/minimind-lab/A01-v2-rl-eligibility-probe-20260908 > "$LOG/probe.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$LOG/probe.pid"
printf 'counterfactual pid=%s; probe pid=%s\n' "$( < "$LOG/counterfactual.pid")" "$( < "$LOG/probe.pid")"
