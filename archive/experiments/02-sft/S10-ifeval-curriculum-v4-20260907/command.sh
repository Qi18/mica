#!/usr/bin/env bash
set -Eeuo pipefail

cd /data/projects/minimind-lab/minimind/trainer
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
torchrun --nproc_per_node=7 train_sft_with_validation.py \
  --save_dir /data/artifacts/minimind-lab/S10-ifeval-curriculum-v4-20260907/checkpoints \
  --save_weight s08_last --best_weight s08_best_val \
  --epochs 3 --batch_size 16 --learning_rate 1e-5 \
  --dtype bfloat16 --num_workers 8 --accumulation_steps 1 --grad_clip 1.0 \
  --log_interval 25 --eval_interval 200 --save_interval 400 \
  --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 --use_moe 0 \
  --train_augment 0 \
  --data_path /data/datasets/minimind-lab/data-v1/sft-ifeval-curriculum-v4/train.jsonl \
  --validation_data_path /data/datasets/minimind-lab/data-v1/sft-ifeval-curriculum-v4/validation.jsonl \
  --from_weight s07r2_best_val \
  --split_manifest /data/artifacts/minimind-lab/S10-ifeval-curriculum-v4-20260907/split/manifest.json \
  --metrics_path /data/artifacts/minimind-lab/S10-ifeval-curriculum-v4-20260907/metrics/metrics.jsonl \
  --use_swanlab --swanlab_project MiniMind-Lab \
  --swanlab_run_name S10-ifeval-curriculum-v4 --use_compile 0
