#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
EXPERIMENT_ID="S03-dense-sft-official-pilot-8m-20260904"
DATA_DIR="/data/datasets/minimind-lab/data-v1/sft-v1-32m/final/official-pilot"
TRAIN_DATA="$DATA_DIR/train.jsonl"
VALIDATION_DATA="$DATA_DIR/validation.jsonl"
TEST_DATA="$DATA_DIR/test.jsonl"
BASE_CHECKPOINT="/data/artifacts/minimind-lab/P03-dense-pretrain-v1-1b28-20260901/formal-b32-a1-epoch1/checkpoints/p03_formal_b32_a1_768.pth"
EXPECTED_BASE_SHA256="0cfb7fc8fd9b3111f30b5528a1c8aacf8d6f633c8cde13c707c7cb44c83fd4fd"
ARTIFACT_DIR="/data/artifacts/minimind-lab/$EXPERIMENT_ID"
MODEL_INPUT_PATH="$ROOT_DIR/minimind/out/pretrain_768.pth"
METRICS_PATH="$ARTIFACT_DIR/metrics/metrics.jsonl"
SPLIT_MANIFEST="$ARTIFACT_DIR/split/validation-file.json"
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

for path in "$TRAIN_DATA" "$VALIDATION_DATA" "$TEST_DATA" "$DATA_DIR/manifest.json" "$DATA_DIR/audit_report.json" "$DATA_DIR/contamination_report.json" "$DATA_DIR/_SUCCESS"; do
  [[ -r "$path" ]] || fail "missing_input:$path"
done
grep -q '"status": "accepted"' "$DATA_DIR/_SUCCESS" || fail dataset_not_accepted
[[ -r "$BASE_CHECKPOINT" ]] || fail base_checkpoint_missing
[[ "$(sha256sum "$BASE_CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_BASE_SHA256" ]] || fail base_checkpoint_sha_mismatch
swanlab verify 2>&1 | grep -q 'logged into https://swanlab.cn as richliu0153' || fail swanlab_not_logged_in
[[ ! -e "$ARTIFACT_DIR" ]] || fail existing_artifact_dir_refuses_overwrite
mkdir -p "$ARTIFACT_DIR/checkpoints" "$ARTIFACT_DIR/logs" "$ARTIFACT_DIR/metrics" "$ARTIFACT_DIR/split" "$ROOT_DIR/minimind/out"
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
  printf 'train_data=%s\n' "$TRAIN_DATA"
  printf 'train_sha256=%s\n' "$(sha256sum "$TRAIN_DATA" | awk '{print $1}')"
  printf 'validation_data=%s\n' "$VALIDATION_DATA"
  printf 'validation_sha256=%s\n' "$(sha256sum "$VALIDATION_DATA" | awk '{print $1}')"
  printf 'test_data=%s\n' "$TEST_DATA"
  printf 'test_sha256=%s\n' "$(sha256sum "$TEST_DATA" | awk '{print $1}')"
  printf 'dataset_manifest_sha256=%s\n' "$(sha256sum "$DATA_DIR/manifest.json" | awk '{print $1}')"
  printf 'trainer_sha256=%s\n' "$(sha256sum "$ROOT_DIR/minimind/trainer/train_sft_with_validation.py" | awk '{print $1}')"
  printf 'swanlab_project=%s\n' MiniMind-Lab
  printf 'swanlab_run_name=%s\n' S03-SFT-OfficialPilot-8M-P03-64M-LR5e-5-B8x8
} > "$RUNTIME_MANIFEST"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits --loop=5 > "$ARTIFACT_DIR/logs/nvidia-smi.csv" &
MONITOR_PID="$!"
cd "$ROOT_DIR/minimind/trainer"
set +e
"$ROOT_DIR/scripts/launch/run_guarded.py" -- python3 -m torch.distributed.run --nproc_per_node=8 train_sft_with_validation.py \
  --epochs 1 --batch_size 8 --accumulation_steps 1 --max_seq_len 768 \
  --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --dtype bfloat16 \
  --learning_rate 5e-5 --grad_clip 1.0 --num_workers 8 --train_augment 0 \
  --log_interval 10 --eval_interval 50 --save_interval 1000 \
  --data_path "$TRAIN_DATA" --validation_data_path "$VALIDATION_DATA" \
  --save_dir "$ARTIFACT_DIR/checkpoints" --save_weight s05a_last \
  --best_weight s05a_best_val --from_weight pretrain \
  --split_seed 42 --split_manifest "$SPLIT_MANIFEST" \
  --metrics_path "$METRICS_PATH" --use_swanlab --swanlab_project MiniMind-Lab \
  --swanlab_run_name S03-SFT-OfficialPilot-8M-P03-64M-LR5e-5-B8x8 2>&1 | tee "$TRAIN_LOG"
rc="$?"
set -e
printf 'finished_at=%s\nexit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >> "$RUNTIME_MANIFEST"
exit "$rc"
