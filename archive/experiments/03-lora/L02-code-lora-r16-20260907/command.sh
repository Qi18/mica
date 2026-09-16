#!/usr/bin/env bash
set -Eeuo pipefail
EXPERIMENT_ID=${EXPERIMENT_ID:-L02-code-lora-r16-20260907}
ROOT_DIR=/data/projects/minimind-lab
ARTIFACT_DIR=/data/artifacts/minimind-lab/$EXPERIMENT_ID
DATA_DIR=/data/datasets/minimind-lab/phase3/code-domain-v1r1
test "$(sha256sum "$ROOT_DIR/minimind/out/s10_best_val_768.pth" | cut -d " " -f1)" = 46aeab66795795aa77d703f08d71b952fe98461040f4560e1301021540710131
for gpu in 2 3 4 5 6 7; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  [[ "$used" -lt 1000 ]] || { echo "target_gpu_busy=$gpu:$used" >&2; exit 1; }
done
grep -q '"status": "accepted"' "$DATA_DIR/manifest.json"
test -s "$DATA_DIR/_SUCCESS"
/usr/local/bin/swanlab verify 2>&1 | grep -q 'logged into https://swanlab.cn as richliu0153'
test ! -e "$ARTIFACT_DIR/checkpoints" || { echo "Existing experiment: use a new EXPERIMENT_ID for replay" >&2; exit 2; }
mkdir -p "$ARTIFACT_DIR"/{checkpoints,logs,metrics,split}
cd "$ROOT_DIR/minimind/trainer"
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
torchrun --standalone --nproc_per_node=6 train_sft_with_validation.py \
  --save_dir "$ARTIFACT_DIR/checkpoints" --save_weight l02_adapter_last --best_weight l02_adapter_best_val \
  --epochs 3 --batch_size 16 --learning_rate 1e-4 --dtype bfloat16 \
  --num_workers 8 --accumulation_steps 1 --grad_clip 1.0 \
  --log_interval 25 --eval_interval 100 --save_interval 300 \
  --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 --use_moe 0 \
  --train_augment 0 --data_path "$DATA_DIR/train.jsonl" \
  --validation_data_path "$DATA_DIR/validation.jsonl" --from_weight s10_best_val \
  --lora_rank 16 --split_seed 42 --split_manifest "$ARTIFACT_DIR/split/manifest.json" \
  --metrics_path "$ARTIFACT_DIR/metrics/metrics.jsonl" \
  --parameter_manifest "$ARTIFACT_DIR/parameters.json" \
  --use_swanlab --swanlab_project MiniMind-Lab \
  --swanlab_run_name L02-Code-LoRA-r16-S10-64M --use_compile 0 \
  2>&1 | tee "$ARTIFACT_DIR/logs/train.log"
