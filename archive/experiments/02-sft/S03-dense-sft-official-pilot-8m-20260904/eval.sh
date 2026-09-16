#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
EXPERIMENT_ID="S03-dense-sft-official-pilot-8m-20260904"
ARTIFACT_DIR="/data/artifacts/minimind-lab/$EXPERIMENT_ID"
CHECKPOINT="$ARTIFACT_DIR/checkpoints/s05a_best_val_768.pth"
EXPORT_DIR="$ARTIFACT_DIR/exported-chat"
BEHAVIOR_DIR="$ARTIFACT_DIR/eval/behavior"
OFFICIAL_DIR="$ARTIFACT_DIR/eval/official-seven"
EVAL_LOG="$ARTIFACT_DIR/eval/eval-driver.log"
SWANLAB_RECEIPT="$ARTIFACT_DIR/eval/swanlab-upload.json"
export CUDA_VISIBLE_DEVICES="0"
export HF_HOME=/data/cache/huggingface
export HF_DATASETS_CACHE=/data/cache/huggingface/datasets
export HF_HUB_CACHE=/data/cache/huggingface/hub
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

[[ -r "$CHECKPOINT" ]] || { printf 'error=checkpoint_missing:%s\n' "$CHECKPOINT" >&2; exit 1; }
[[ ! -e "$EXPORT_DIR" ]] || { printf 'error=export_dir_exists:%s\n' "$EXPORT_DIR" >&2; exit 1; }
[[ ! -e "$ARTIFACT_DIR/eval" ]] || { printf 'error=eval_dir_exists:%s\n' "$ARTIFACT_DIR/eval" >&2; exit 1; }
mkdir -p "$ARTIFACT_DIR/eval" "$BEHAVIOR_DIR" "$OFFICIAL_DIR"

{
  printf 'evaluation_started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'experiment_id=%s\n' "$EXPERIMENT_ID"
  printf 'checkpoint=%s\n' "$CHECKPOINT"
  printf 'checkpoint_sha256=%s\n' "$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
  printf 'physical_gpu=%s\n' 0

  cd "$ROOT_DIR/minimind/scripts"
  python3 -c "import torch; import convert_model as c; c.lm_config=c.MiniMindConfig(hidden_size=768,num_hidden_layers=8,max_seq_len=8192,use_moe=False); c.convert_torch2transformers('$CHECKPOINT','$EXPORT_DIR',torch.float16)"

  cd "$ROOT_DIR"
  python3 scripts/eval/eval_sft_behavior.py \
    --model "$EXPORT_DIR" --output-dir "$BEHAVIOR_DIR" \
    --device cuda:0 --seed 42 --max-new-tokens 192

  /data/venvs/minimind-eval/bin/lm_eval run \
    --model hf \
    --model_args "pretrained=$EXPORT_DIR,dtype=auto,trust_remote_code=True" \
    --tasks ceval-valid,cmmlu,arc_easy,piqa,openbookqa,hellaswag,social_iqa \
    --num_fewshot 0 --batch_size 16 --device cuda:0 --seed 42 \
    --output_path "$OFFICIAL_DIR/results.json"
  RESULT_JSON="$(find "$OFFICIAL_DIR" -maxdepth 1 -type f -name 'results_*.json' -print -quit)"
  [[ -n "$RESULT_JSON" ]] || { printf 'error=official_result_missing\n' >&2; exit 1; }


  python3 scripts/eval/log_eval_to_swanlab.py \
    --official "$RESULT_JSON" \
    --behavior "$BEHAVIOR_DIR/task_eval.json" \
    --system "$BEHAVIOR_DIR/system_metrics.json" \
    --project MiniMind-Lab --experiment-name "S03-Eval-OfficialPilot-8M-P03-64M" --group S03-S04-equal-budget \
    --output "$SWANLAB_RECEIPT"

  printf 'evaluation_finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$EVAL_LOG"
