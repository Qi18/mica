#!/usr/bin/env bash
set -euo pipefail
cd /data/projects/minimind-lab
PY=/data/venvs/minimind-eval/bin/python
case "${1:-}" in
 A00)
 MODEL=/data/artifacts/minimind-lab/D01-s10-preference-baseline-20260908/exported-fp32
 OUT=/data/artifacts/minimind-lab/A00-v2-frozen-test-20260908
 ;;
 A01)
 MODEL=/data/artifacts/minimind-lab/A01-agent-sft-v2-pilot-20260908/checkpoint-225
 OUT=/data/artifacts/minimind-lab/A01-v2-frozen-test-20260908
 ;;
 *) exit 2 ;;
esac
"$PY" -u scripts/eval/eval_phase6_agent.py --model "$MODEL" \
 --tasks /data/datasets/minimind-lab/phase6/record-agent-v2/test.jsonl --environment phase6_agent_v2 \
 --output "$OUT/agent-test" --run-name "$1-v2-Frozen-Agent-Test-20260908"
"$PY" -u scripts/eval/eval_sft_behavior.py --model "$MODEL" --output-dir "$OUT/behavior" --max-new-tokens 128
