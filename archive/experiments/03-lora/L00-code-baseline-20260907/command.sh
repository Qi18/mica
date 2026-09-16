#!/usr/bin/env bash
set -Eeuo pipefail
export CUDA_VISIBLE_DEVICES=3
export HF_ALLOW_CODE_EVAL=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/cache/huggingface
export TOKENIZERS_PARALLELISM=false
/data/venvs/minimind-eval/bin/lm_eval run \
  --model hf \
  --model_args pretrained=/data/artifacts/minimind-lab/S10-ifeval-curriculum-v4-20260907/exported-best,dtype=float16,trust_remote_code=True \
  --tasks mbpp_instruct --num_fewshot 0 --batch_size 16 --device cuda:0 \
  --apply_chat_template --seed 42 --confirm_run_unsafe_code --log_samples \
  --output_path /data/artifacts/minimind-lab/L00-code-baseline-20260907/eval/mbpp-full/results.json
