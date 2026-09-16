#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
EXPERIMENT_DIR="$ROOT_DIR/experiments/00-preparation/E01-tokenizer-dataset-20260823"
PYTHON_BIN="${PYTHON_BIN:-/data/venvs/minimind-lab/bin/python}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/data/cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1

cd "$ROOT_DIR"
"$PYTHON_BIN" scripts/launch/preflight_l20.py
"$PYTHON_BIN" scripts/sync/download_minimind_dataset.py \
  --manifest "$EXPERIMENT_DIR/data_manifest.json"
"$PYTHON_BIN" scripts/eval/analyze_tokenizer_dataset.py \
  --manifest "$EXPERIMENT_DIR/data_manifest.json" \
  --fixture-dir "$EXPERIMENT_DIR/fixtures" \
  --output-dir "$EXPERIMENT_DIR/results" \
  --max-length 768 \
  --fixture-max-length 128 \
  --audit-sample-size 2000 \
  --domain-samples 20 \
  --seed 42
"$PYTHON_BIN" scripts/sync/validate_experiment.py "$EXPERIMENT_DIR"
