#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
EXPERIMENT_DIR="$ROOT_DIR/experiments/00-preparation/D02-sft-data-v1-20260901"
CONFIG="$ROOT_DIR/configs/data/sft/sources_v1.yaml"
PYTHON_BIN="${PYTHON_BIN:-/data/venvs/minimind-lab/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/datasets/minimind-lab/data-v1/sft-v1-160m}"
RUN_MODE="${RUN_MODE:-validate}"
RESOLVED="$EXPERIMENT_DIR/resolved_sources.json"
RAW_MANIFEST="$EXPERIMENT_DIR/raw_materialization_manifest.json"

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="${HF_HOME:-/data/cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

run_low_priority() {
  if command -v ionice >/dev/null 2>&1; then
    nice -n 10 ionice -c 2 -n 7 "$@"
  else
    nice -n 10 "$@"
  fi
}

acquire_materialize_lock() {
  local lock_file="$DATA_ROOT/.materialize.lock"
  command -v flock >/dev/null 2>&1 || {
    printf 'error=missing_command:flock\n' >&2
    exit 69
  }
  mkdir -p "$DATA_ROOT"
  exec 9>"$lock_file"
  if ! flock -n 9; then
    printf 'error=materialization_already_running lock=%s\n' "$lock_file" >&2
    exit 75
  fi
  printf '%s\n' "$$" >&9
}

case "$RUN_MODE" in
  self-test)
    "$PYTHON_BIN" -m py_compile "$ROOT_DIR/scripts/data/sft/materialize_raw.py"
    bash -n "$EXPERIMENT_DIR/command.sh"
    run_low_priority "$PYTHON_BIN" "$ROOT_DIR/scripts/data/sft/materialize_raw.py" --self-test
    ;;
  resolve)
    run_low_priority "$PYTHON_BIN" "$ROOT_DIR/scripts/data/sft/resolve_sources.py" \
      --config "$CONFIG" \
      --endpoint "$HF_ENDPOINT" \
      --timeout 30 \
      --output "$RESOLVED"
    ;;
  validate)
    run_low_priority "$PYTHON_BIN" "$ROOT_DIR/scripts/data/sft/materialize_raw.py" \
      --config "$CONFIG" \
      --resolved-sources "$RESOLVED" \
      --output-root "$DATA_ROOT" \
      --manifest "$RAW_MANIFEST" \
      --validate-only
    ;;
  materialize)
    acquire_materialize_lock
    run_low_priority "$PYTHON_BIN" "$ROOT_DIR/scripts/data/sft/resolve_sources.py" \
      --config "$CONFIG" \
      --endpoint "$HF_ENDPOINT" \
      --timeout 30 \
      --output "$RESOLVED"
    run_low_priority "$PYTHON_BIN" "$ROOT_DIR/scripts/data/sft/materialize_raw.py" \
      --config "$CONFIG" \
      --resolved-sources "$RESOLVED" \
      --output-root "$DATA_ROOT" \
      --manifest "$RAW_MANIFEST"
    ;;
  *)
    printf 'error=unsupported_RUN_MODE:%s\n' "$RUN_MODE" >&2
    exit 2
    ;;
esac
