#!/usr/bin/env bash
# Replay a Phase3 candidate evaluation into a NEW artifact directory.
set -Eeuo pipefail
exp=${1:?Usage: run_phase3_eval.sh EXPERIMENT_ID PHYSICAL_GPU NEW_OUTPUT_DIR}
gpu=${2:?}
output=${3:?}
case "$exp" in L00-code-baseline-20260907) model=/data/artifacts/minimind-lab/S10-ifeval-curriculum-v4-20260907/exported-best;;
L01-code-full-ft-20260907|L02-code-lora-r16-20260907) model=/data/artifacts/minimind-lab/$exp/exported-best;;
*) echo "Unknown Phase3 experiment" >&2; exit 2;; esac
test ! -e "$output" || { echo "Output exists; choose a new directory" >&2; exit 2; }
used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
test "$used" -lt 1000 || { echo "Target GPU busy" >&2; exit 2; }
cd /data/projects/minimind-lab
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES="$gpu" HF_HOME=/data/cache/huggingface HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for mode in mbpp official-seven-chat-template ifeval-chat-template; do
  extra=()
  case "$mode" in
    mbpp) tasks=mbpp_instruct; extra=(--confirm_run_unsafe_code --log_samples); export HF_ALLOW_CODE_EVAL=1;;
    official-seven-chat-template) tasks=ceval-valid,cmmlu,arc_easy,piqa,openbookqa,hellaswag,social_iqa;;
    ifeval-chat-template) tasks=ifeval; extra=(--log_samples);;
  esac
  mkdir -p "$output/$mode"
  /data/venvs/minimind-eval/bin/lm_eval --model hf --model_args "pretrained=$model,dtype=float16" \
    --tasks "$tasks" --num_fewshot 0 --batch_size 16 --apply_chat_template --seed 42 \
    "${extra[@]}" --output_path "$output/$mode" 2>&1 | tee "$output/$mode/eval.log"
done
/data/venvs/minimind-eval/bin/python scripts/eval/eval_sft_behavior.py --model "$model" --output-dir "$output/behavior"
