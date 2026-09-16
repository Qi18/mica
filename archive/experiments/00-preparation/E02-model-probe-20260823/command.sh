#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
EXPERIMENT_DIR="$ROOT_DIR/experiments/00-preparation/E02-model-probe-20260823"
PYTHON_BIN="${PYTHON_BIN:-/data/venvs/minimind-lab/bin/python}"
DATA_REVISION="312afb4f76391145c6902f765bb51691c09a12f5"
SOURCE_DATA="/data/datasets/minimind/$DATA_REVISION/pretrain_t2t_mini.jsonl"
SOURCE_SHA256="6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c"
ARTIFACT_DIR="/data/artifacts/minimind-lab/E02-model-probe-20260823"
PROBE_DATA="$ARTIFACT_DIR/probe-data.pt"
CHECKPOINT="$ARTIFACT_DIR/checkpoints/seed42-resume.pt"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LAB_COMMIT="$(git -C "$ROOT_DIR" rev-parse HEAD)"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/results" "$ARTIFACT_DIR/checkpoints"
cd "$ROOT_DIR"

"$PYTHON_BIN" scripts/launch/preflight_l20.py
"$PYTHON_BIN" scripts/eval/prepare_model_probe_data.py \
  --source "$SOURCE_DATA" \
  --source-sha256 "$SOURCE_SHA256" \
  --tokenizer minimind/model \
  --output "$PROBE_DATA" \
  --manifest "$EXPERIMENT_DIR/results/probe-data-manifest.json" \
  --samples 8192 \
  --max-length 128

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/eval/inspect_model_architecture.py \
  --output "$EXPERIMENT_DIR/results/architecture.json" \
  --batch-size 2 \
  --seq-len 16 \
  2>&1 | tee "$EXPERIMENT_DIR/logs/architecture.log"

run_probe() {
  local seed="$1"
  local max_steps="$2"
  local resume_flag="${3:-}"
  local checkpoint_args=()
  if [[ "$seed" == "42" ]]; then
    checkpoint_args=(--checkpoint "$CHECKPOINT" --save-every 50)
  fi
  "$PYTHON_BIN" scripts/launch/run_guarded.py -- \
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/launch/train_model_probe.py \
      --data "$PROBE_DATA" \
      --metrics "$EXPERIMENT_DIR/results/metrics-seed${seed}.jsonl" \
      --summary "$EXPERIMENT_DIR/results/summary-seed${seed}.json" \
      --run-id-file "$EXPERIMENT_DIR/results/swanlab-seed${seed}.txt" \
      --seed "$seed" \
      --max-steps "$max_steps" \
      --total-steps 100 \
      --batch-size 4 \
      --learning-rate 5e-4 \
      --log-interval 10 \
      "${checkpoint_args[@]}" \
      $resume_flag
}

run_probe 42 50 2>&1 | tee "$EXPERIMENT_DIR/logs/seed42-phase1.log"
run_probe 42 100 --resume 2>&1 | tee "$EXPERIMENT_DIR/logs/seed42-resume.log"
run_probe 43 100 2>&1 | tee "$EXPERIMENT_DIR/logs/seed43.log"
run_probe 44 100 2>&1 | tee "$EXPERIMENT_DIR/logs/seed44.log"

"$PYTHON_BIN" scripts/eval/summarize_model_probe.py \
  --experiment-dir "$EXPERIMENT_DIR" \
  --checkpoint "$CHECKPOINT" \
  --lab-commit "$LAB_COMMIT" \
  --started-at "$STARTED_AT"
"$PYTHON_BIN" scripts/sync/validate_experiment.py "$EXPERIMENT_DIR"
