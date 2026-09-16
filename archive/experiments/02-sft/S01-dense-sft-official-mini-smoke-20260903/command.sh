#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
EXPERIMENT_ID="S01-dense-sft-official-mini-smoke-20260903"
EXPERIMENT_DIR="$ROOT_DIR/experiments/02-sft/$EXPERIMENT_ID"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_DATA="/data/datasets/minimind/312afb4f76391145c6902f765bb51691c09a12f5/sft_t2t_mini.jsonl"
DATA_ROOT="/data/datasets/minimind-lab/phase2/$EXPERIMENT_ID"
DATA_PATH="$DATA_ROOT/train-validation.jsonl"
DATA_MANIFEST="$DATA_ROOT/manifest.json"
BASE_CHECKPOINT="/data/artifacts/minimind-lab/P03-dense-pretrain-v1-1b28-20260901/formal-b32-a1-epoch1/checkpoints/p03_formal_b32_a1_768.pth"
EXPECTED_BASE_SHA256="0cfb7fc8fd9b3111f30b5528a1c8aacf8d6f633c8cde13c707c7cb44c83fd4fd"
ARTIFACT_DIR="/data/artifacts/minimind-lab/$EXPERIMENT_ID"
MODEL_INPUT_PATH="$ROOT_DIR/minimind/out/pretrain_768.pth"
METRICS_PATH="$ARTIFACT_DIR/metrics/metrics.jsonl"
SPLIT_MANIFEST="$ARTIFACT_DIR/split/validation-indices.json"
TRAIN_LOG="$ARTIFACT_DIR/logs/train.log"
RUNTIME_MANIFEST="$ARTIFACT_DIR/runtime-manifest.txt"
MONITOR_PID=""

fail() { printf 'error=%s\n' "$1" >&2; exit 1; }
cleanup() {
  if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM HUP

[[ -r "$SOURCE_DATA" ]] || fail source_data_missing
[[ -r "$BASE_CHECKPOINT" ]] || fail base_checkpoint_missing
[[ "$(sha256sum "$BASE_CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_BASE_SHA256" ]] || fail base_checkpoint_sha_mismatch
swanlab verify 2>&1 | grep -q 'logged into https://swanlab.cn as richliu0153' || fail swanlab_not_logged_in
mkdir -p "$DATA_ROOT" "$ARTIFACT_DIR/checkpoints" "$ARTIFACT_DIR/logs" "$ARTIFACT_DIR/metrics" "$ARTIFACT_DIR/split" "$ROOT_DIR/minimind/out"
[[ ! -e "$ARTIFACT_DIR/checkpoints/s01_last_768.pth" ]] || fail existing_checkpoint_refuses_overwrite

TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  "$PYTHON_BIN" "$EXPERIMENT_DIR/prepare_smoke_dataset.py" \
  --input "$SOURCE_DATA" --output "$DATA_PATH" --manifest "$DATA_MANIFEST" \
  --tokenizer "$ROOT_DIR/minimind/model" --rows 3000 --validation-size 256 --seed 42 --max-length 768

ln -sfn "$BASE_CHECKPOINT" "$MODEL_INPUT_PATH"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/data/cache/huggingface HF_DATASETS_CACHE=/data/cache/huggingface/datasets HF_HUB_CACHE=/data/cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
{
  printf 'experiment_id=%s\n' "$EXPERIMENT_ID"
  printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'lab_commit=%s\n' "$(git -C "$ROOT_DIR" rev-parse HEAD)"
  printf 'working_tree_clean=%s\n' "$(test -z "$(git -C "$ROOT_DIR" status --porcelain)" && echo true || echo false)"
  printf 'base_checkpoint=%s\n' "$BASE_CHECKPOINT"
  printf 'base_checkpoint_sha256=%s\n' "$EXPECTED_BASE_SHA256"
  printf 'data_path=%s\n' "$DATA_PATH"
  printf 'data_sha256=%s\n' "$(sha256sum "$DATA_PATH" | awk '{print $1}')"
  printf 'data_manifest=%s\n' "$DATA_MANIFEST"
  printf 'trainer_sha256=%s\n' "$(sha256sum "$ROOT_DIR/minimind/trainer/train_sft_with_validation.py" | awk '{print $1}')"
  printf 'swanlab_project=%s\n' MiniMind-Lab
  printf 'swanlab_run_name=%s\n' S01-SFT-OfficialMini-Smoke-P03-64M-B8x8
} > "$RUNTIME_MANIFEST"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits --loop=5 > "$ARTIFACT_DIR/logs/nvidia-smi.csv" &
MONITOR_PID="$!"
cd "$ROOT_DIR/minimind/trainer"
set +e
"$ROOT_DIR/scripts/launch/run_guarded.py" -- "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node=8 train_sft_with_validation.py \
  --epochs 1 --batch_size 8 --accumulation_steps 1 --max_seq_len 768 \
  --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --dtype bfloat16 \
  --learning_rate 1e-5 --grad_clip 1.0 --num_workers 8 --train_augment 0 \
  --log_interval 5 --eval_interval 10 --save_interval 20 \
  --data_path "$DATA_PATH" --save_dir "$ARTIFACT_DIR/checkpoints" \
  --save_weight s01_last --best_weight s01_best_val --from_weight pretrain \
  --validation_size 256 --split_seed 42 --split_manifest "$SPLIT_MANIFEST" \
  --metrics_path "$METRICS_PATH" --use_swanlab --swanlab_project MiniMind-Lab \
  --swanlab_run_name S01-SFT-OfficialMini-Smoke-P03-64M-B8x8 2>&1 | tee "$TRAIN_LOG"
rc="${PIPESTATUS[0]}"
set -e
printf 'finished_at=%s\nexit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >> "$RUNTIME_MANIFEST"
exit "$rc"
