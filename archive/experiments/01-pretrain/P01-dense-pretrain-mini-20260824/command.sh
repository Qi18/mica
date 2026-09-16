#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
EXPERIMENT_ID="P01-dense-pretrain-mini-20260824"
PYTHON_BIN="${PYTHON_BIN:-/data/venvs/minimind-lab/bin/python}"
DATA_PATH="${DATA_PATH:-/data/datasets/minimind/312afb4f76391145c6902f765bb51691c09a12f5/pretrain_t2t_mini.jsonl}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/data/artifacts/minimind-lab/$EXPERIMENT_ID}"
FROM_RESUME="${FROM_RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
EXPECTED_DATA_SHA256="6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c"
SAVE_WEIGHT="p01_pretrain"
RESUME_PATH="$ROOT_DIR/minimind/checkpoints/${SAVE_WEIGHT}_768_resume.pth"
HARDWARE_LOG="$ARTIFACT_DIR/logs/nvidia-smi.csv"
RUNTIME_MANIFEST="$ARTIFACT_DIR/runtime-manifest.txt"
MONITOR_PID=""

fail() {
  printf 'error=%s\n' "$1" >&2
  exit 1
}

cleanup() {
  if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM HUP

[[ -x "$PYTHON_BIN" ]] || fail "python_not_executable:$PYTHON_BIN"
[[ -r "$DATA_PATH" ]] || fail "dataset_not_readable:$DATA_PATH"
[[ "$FROM_RESUME" == "0" || "$FROM_RESUME" == "1" ]] || fail "FROM_RESUME_must_be_0_or_1"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fail "DRY_RUN_must_be_0_or_1"

if [[ "$FROM_RESUME" == "1" ]]; then
  [[ -r "$RESUME_PATH" ]] || fail "resume_checkpoint_not_found:$RESUME_PATH"
elif [[ -e "$RESUME_PATH" ]]; then
  fail "stale_resume_checkpoint_exists:$RESUME_PATH"
fi

actual_sha256="$(sha256sum "$DATA_PATH" | awk '{print $1}')"
[[ "$actual_sha256" == "$EXPECTED_DATA_SHA256" ]] || fail "dataset_sha256_mismatch:$actual_sha256"

mkdir -p "$ARTIFACT_DIR/checkpoints" "$ARTIFACT_DIR/logs"
{
  printf 'experiment_id=%s\n' "$EXPERIMENT_ID"
  printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'lab_commit=%s\n' "$(git -C "$ROOT_DIR" rev-parse HEAD)"
  printf 'minimind_source_commit=%s\n' "393e387e9ad99f0f04c296e4c5e7353f4444629f"
  printf 'dataset_path=%s\n' "$DATA_PATH"
  printf 'dataset_sha256=%s\n' "$actual_sha256"
  printf 'artifact_dir=%s\n' "$ARTIFACT_DIR"
  printf 'from_resume=%s\n' "$FROM_RESUME"
} > "$RUNTIME_MANIFEST"

nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits \
  --loop=10 > "$HARDWARE_LOG" &
MONITOR_PID="$!"

GUARD_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
  GUARD_ARGS+=(--dry-run)
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "$ROOT_DIR/minimind/trainer"

"$ROOT_DIR/scripts/launch/run_guarded.py" "${GUARD_ARGS[@]}" -- \
  "$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node=8 \
  train_pretrain.py \
  --epochs 1 \
  --batch_size 4 \
  --accumulation_steps 8 \
  --max_seq_len 768 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --dtype bfloat16 \
  --learning_rate 5e-4 \
  --grad_clip 1.0 \
  --num_workers 8 \
  --log_interval 10 \
  --save_interval 1000 \
  --data_path "$DATA_PATH" \
  --save_dir "$ARTIFACT_DIR/checkpoints" \
  --save_weight "$SAVE_WEIGHT" \
  --from_weight none \
  --from_resume "$FROM_RESUME" \
  --use_wandb \
  --wandb_project MiniMind-Lab-Stage3
