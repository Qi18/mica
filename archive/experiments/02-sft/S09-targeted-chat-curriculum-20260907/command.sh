#!/usr/bin/env bash
set -Eeuo pipefail

cd /data/projects/minimind-lab/minimind/trainer
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
torchrun --nproc_per_node=7 train_sft_with_validation.py \
  --save_dir /data/artifacts/minimind-lab/S09-targeted-chat-curriculum-20260907/checkpoints \
  --save_weight s07r2_last --best_weight s07r2_best_val \
  --epochs 5 --batch_size 16 --learning_rate 8e-6 \
  --dtype bfloat16 --num_workers 8 --accumulation_steps 1 --grad_clip 1.0 \
  --log_interval 50 --eval_interval 400 --save_interval 800 \
  --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 --use_moe 0 \
  --train_augment 1 \
  --data_path /data/datasets/minimind-lab/data-v1/sft-chat-repair-v3/train.jsonl \
  --validation_data_path /data/datasets/minimind-lab/data-v1/sft-chat-repair-v3/validation.jsonl \
  --from_weight s07r1_best_val \
  --split_manifest /data/artifacts/minimind-lab/S09-targeted-chat-curriculum-20260907/split/manifest.json \
  --metrics_path /data/artifacts/minimind-lab/S09-targeted-chat-curriculum-20260907/metrics/metrics.jsonl \
  --use_swanlab --swanlab_project MiniMind-Lab \
  --swanlab_run_name S09-targeted-chat-curriculum --use_compile 0
