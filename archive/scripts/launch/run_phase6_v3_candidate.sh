#!/usr/bin/env bash
set -euo pipefail
cd /data/projects/minimind-lab
PY=/data/venvs/minimind-eval/bin/python
LR="${1:?lr3e6 or lr1e6}"
case "$LR" in lr3e6|lr1e6) ;; *) exit 2;; esac
OUT="/data/artifacts/minimind-lab/A01-v3-mixed-$LR-20260908"
"$PY" -u scripts/launch/train_phase6_sft.py \
 --config "configs/agentic_rl/phase6-a01-v3-$LR.json" --output "$OUT" \
 --run-name "A01-v3-Mixed-$LR-20260908"
MODEL=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$OUT/best-checkpoint.json")
"$PY" -u scripts/eval/eval_phase6_agent.py --model "$MODEL" \
 --tasks /data/datasets/minimind-lab/phase6/mixed-agent-v3-r1/tools-val.jsonl \
 --environment phase6_retention_env --output "$OUT/validation-tools" --run-name "A01-v3-$LR-Tools-Val-20260908"
"$PY" -u scripts/eval/eval_sft_behavior.py --model "$MODEL" --output-dir "$OUT/behavior" --max-new-tokens 128
