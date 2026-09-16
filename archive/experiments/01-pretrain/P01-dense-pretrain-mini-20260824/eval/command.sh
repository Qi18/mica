#!/usr/bin/env bash
set -euo pipefail

LAB_DIR=/data/projects/minimind-lab
EXP_DIR="$LAB_DIR/experiments/01-pretrain/P01-dense-pretrain-mini-20260824"
ARTIFACT_DIR=/data/artifacts/minimind-lab/P01-dense-pretrain-mini-20260824
CHECKPOINT="$ARTIFACT_DIR/checkpoints/p01_pretrain_768.pth"
EXPORT_DIR="$ARTIFACT_DIR/exported-base"
EVAL_VENV=/data/venvs/minimind-eval

cd "$LAB_DIR"
mkdir -p "$EXP_DIR/eval" "$ARTIFACT_DIR/eval/official"

"$EVAL_VENV/bin/python" scripts/eval/export_minimind_base.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$EXPORT_DIR" \
  --minimind-dir minimind \
  --hidden-size 768 \
  --num-hidden-layers 8 \
  --dtype float16

"$EVAL_VENV/bin/python" scripts/eval/eval_base_continuation.py \
  --model "$EXPORT_DIR" \
  --output "$EXP_DIR/eval/samples.jsonl" \
  --device cuda:0 \
  --max-new-tokens 64

"$EVAL_VENV/bin/python" scripts/eval/summarize_pretrain.py \
  --driver-log "$ARTIFACT_DIR/driver.log" \
  --gpu-log "$ARTIFACT_DIR/logs/nvidia-smi.csv" \
  --checkpoint "$CHECKPOINT" \
  --started-at 2026-08-24T03:16:40Z \
  --world-size 8 \
  --per-gpu-batch-size 4 \
  --max-seq-len 768 \
  --output "$EXP_DIR/eval/system_metrics.json"

HF_HOME=/data/cache/huggingface \
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
CUDA_VISIBLE_DEVICES=0 \
"$EVAL_VENV/bin/lm-eval" run \
  --model hf \
  --model_args "pretrained=$EXPORT_DIR" dtype=auto trust_remote_code=True \
  --tasks ceval-valid cmmlu arc_easy piqa openbookqa hellaswag social_iqa \
  --batch_size 16 \
  --device cuda:0 \
  --trust_remote_code \
  --seed 42 \
  --output_path "$ARTIFACT_DIR/eval/official" \
  2>&1 | tee "$ARTIFACT_DIR/eval/official.log"
