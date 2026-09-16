#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
EXPERIMENT_DIR="$ROOT_DIR/experiments/00-preparation/D03-sft-capacity-20260901"
PROFILE_CONFIG="$EXPERIMENT_DIR/config.json"
PROFILER="$ROOT_DIR/experiments/00-preparation/D05-sft-capacity-v2-20260901/tools/profile_sft_v1_capacity.py"
TOKENIZER="$ROOT_DIR/minimind/model"
OUTPUT="$EXPERIMENT_DIR/capacity_report.json"
PYTHON_BIN="${PYTHON_BIN:-/data/venvs/minimind-lab/bin/python}"
RUN_MODE="${RUN_MODE:-validate}"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0

run_low_priority() {
  if command -v ionice >/dev/null 2>&1; then
    nice -n 10 ionice -c 2 -n 7 "$@"
  else
    nice -n 10 "$@"
  fi
}

case "$RUN_MODE" in
  self-test)
    "$PYTHON_BIN" -m py_compile "$PROFILER"
    bash -n "$EXPERIMENT_DIR/command.sh"
    run_low_priority "$PYTHON_BIN" "$PROFILER" \
      --self-test \
      --tokenizer "$TOKENIZER"
    ;;
  validate)
    run_low_priority "$PYTHON_BIN" "$PROFILER" \
      --profile-config "$PROFILE_CONFIG" \
      --validate-only
    ;;
  profile)
    run_low_priority "$PYTHON_BIN" "$PROFILER" \
      --profile-config "$PROFILE_CONFIG" \
      --output "$OUTPUT"
    ;;
  *)
    printf "error=unsupported_RUN_MODE:%s\n" "$RUN_MODE" >&2
    exit 2
    ;;
esac
