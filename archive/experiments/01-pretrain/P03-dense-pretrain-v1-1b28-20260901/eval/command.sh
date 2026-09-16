#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXPERIMENT_ID="P03-dense-pretrain-v1-1b28-20260901"
EXP_DIR="$ROOT_DIR/experiments/01-pretrain/$EXPERIMENT_ID"
EVAL_DIR="$EXP_DIR/eval"
STEP_DIR="$EVAL_DIR/.steps"
DATA_ROOT="/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/final-remix-v1"
VALIDATION_FILE="$DATA_ROOT/validation-00000-of-00001.jsonl"
SUCCESS_MARKER="$DATA_ROOT/_SUCCESS"
EXPECTED_DATASET_FINGERPRINT="cd018f6d0a047284f5f77d240d2583a1673c9d9a923536e9da7e4b1e4ead70bd"
EXPECTED_VALIDATION_ROWS=11525
EXPECTED_VALIDATION_TARGET_TOKENS=6400000

LAB_PYTHON="${LAB_PYTHON:-/data/venvs/minimind-lab/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/data/venvs/minimind-eval/bin/python}"
LM_EVAL="${LM_EVAL:-/data/venvs/minimind-eval/bin/lm-eval}"
DRY_RUN="${DRY_RUN:-}"

P01_CHECKPOINT="/data/artifacts/minimind-lab/P01-dense-pretrain-mini-20260824/checkpoints/p01_pretrain_768.pth"
P02_CHECKPOINT="/data/artifacts/minimind-lab/P02-dense-pretrain-full-20260824/checkpoints/p02_pretrain_768.pth"
P03_RUN_DIR="${P03_RUN_DIR:-/data/artifacts/minimind-lab/$EXPERIMENT_ID/formal-b32-a1-epoch1}"
P03_CHECKPOINT="${P03_CHECKPOINT:-$P03_RUN_DIR/checkpoints/p03_formal_b32_a1_768.pth}"
P03_EXIT_STATUS="${P03_EXIT_STATUS:-$P03_RUN_DIR/exit-status.json}"
P03_RUNTIME_MANIFEST="${P03_RUNTIME_MANIFEST:-$P03_RUN_DIR/runtime-manifest.json}"
P03_DRIVER_LOG="${P03_DRIVER_LOG:-$P03_RUN_DIR/driver.log}"
P03_GPU_LOG="${P03_GPU_LOG:-$P03_RUN_DIR/logs/nvidia-smi.csv}"
P03_EXPORT_DIR="${P03_EXPORT_DIR:-$P03_RUN_DIR/exported-base}"

P01_VALIDATION="$EVAL_DIR/validation-p01-exact.json"
P02_VALIDATION="$EVAL_DIR/validation-p02-exact.json"
P03_VALIDATION="$EVAL_DIR/validation-p03-exact.json"
CONTINUATIONS="$EVAL_DIR/continuations-p03-5.jsonl"
OFFICIAL_JSON="$EVAL_DIR/official-seven-p03.json"
OFFICIAL_LOG="$EVAL_DIR/official-seven-p03.log"
LATENCY_JSON="$EVAL_DIR/latency-p03-warmup5-measure20.json"
SYSTEM_JSON="$EVAL_DIR/system-p03.json"
SWANLAB_RECEIPT="$EVAL_DIR/swanlab-p03-eval.json"

fail() {
  printf 'error=%s\n' "$1" >&2
  exit 1
}

require_file() {
  [[ -r "$1" ]] || fail "required_file_not_readable:$1"
}

print_command() {
  local step="$1"
  shift
  printf 'PLAN step=%s command=' "$step"
  printf '%q ' "$@"
  printf '\n'
}

step_manifest() {
  printf '%s/%s.json' "$STEP_DIR" "$1"
}

step_reusable() {
  local name="$1" input_count="$2" output_count="$3"
  shift 3
  local manifest
  manifest="$(step_manifest "$name")"
  local paths=("$@")
  local outputs=("${paths[@]:input_count:output_count}")
  if [[ ! -e "$manifest" ]]; then
    local output
    for output in "${outputs[@]}"; do
      [[ ! -e "$output" ]] || fail "stale_output_without_step_manifest:$name:$output"
    done
    return 1
  fi
  "$LAB_PYTHON" - "$manifest" "$name" "$input_count" "$output_count" "${paths[@]}" <<'PY' || exit 1
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
name = sys.argv[2]
input_count = int(sys.argv[3])
output_count = int(sys.argv[4])
paths = [Path(item).resolve() for item in sys.argv[5:]]
inputs = paths[:input_count]
outputs = paths[input_count:input_count + output_count]

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = json.loads(manifest_path.read_text(encoding="utf-8"))
if payload.get("status") != "completed" or payload.get("step") != name:
    raise SystemExit(f"invalid_step_manifest={manifest_path}")
for key, expected in (("inputs", inputs), ("outputs", outputs)):
    recorded = payload.get(key)
    expected_paths = [str(path) for path in expected]
    if not isinstance(recorded, list) or [item.get("path") for item in recorded] != expected_paths:
        raise SystemExit(f"step_{key}_path_mismatch={name}")
    for item, path in zip(recorded, expected):
        if not path.is_file() or item.get("sha256") != sha(path):
            raise SystemExit(f"step_{key}_hash_mismatch={name}:{path}")
print(f"REUSE step={name} manifest={manifest_path}")
PY
}

mark_step() {
  local name="$1" input_count="$2" output_count="$3"
  shift 3
  local manifest
  manifest="$(step_manifest "$name")"
  mkdir -p "$STEP_DIR"
  "$LAB_PYTHON" - "$manifest" "$name" "$input_count" "$output_count" "$@" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest = Path(sys.argv[1])
name = sys.argv[2]
input_count = int(sys.argv[3])
output_count = int(sys.argv[4])
paths = [Path(item).resolve() for item in sys.argv[5:]]
inputs = paths[:input_count]
outputs = paths[input_count:input_count + output_count]
if len(paths) != input_count + output_count:
    raise SystemExit("step_path_count_mismatch")

def describe(path):
    if not path.is_file():
        raise SystemExit(f"step_file_missing={path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}

payload = {
    "status": "completed",
    "step": name,
    "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "inputs": [describe(path) for path in inputs],
    "outputs": [describe(path) for path in outputs],
}
manifest.parent.mkdir(parents=True, exist_ok=True)
temporary = manifest.with_name(f".{manifest.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, manifest)
print(f"DONE step={name} manifest={manifest}")
PY
}

run_step() {
  local name="$1" input_count="$2" output_count="$3"
  shift 3
  local path_count=$((input_count + output_count))
  local paths=("${@:1:path_count}")
  shift "$path_count"
  [[ "${1:-}" == "--" ]] || fail "step_command_separator_missing:$name"
  shift
  local command=("$@")
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$name" "${command[@]}"
    return 0
  fi
  if step_reusable "$name" "$input_count" "$output_count" "${paths[@]}"; then
    return 0
  fi
  "${command[@]}"
  mark_step "$name" "$input_count" "$output_count" "${paths[@]}"
}

verify_formal_training_exit() {
  "$LAB_PYTHON" - "$P03_EXIT_STATUS" "$P03_CHECKPOINT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

status_path, checkpoint_path = map(Path, sys.argv[1:])
payload = json.loads(status_path.read_text(encoding="utf-8"))
checkpoint = checkpoint_path.resolve()
if payload.get("exit_code") != 0 or payload.get("checkpoint_exists") is not True:
    raise SystemExit("p03_formal_training_not_exit0")
if Path(payload.get("checkpoint", "")).resolve() != checkpoint:
    raise SystemExit("p03_exit_status_checkpoint_path_mismatch")
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if payload.get("checkpoint_sha256") != digest:
    raise SystemExit("p03_exit_status_checkpoint_sha_mismatch")
print(f"formal_training_gate=pass checkpoint_sha256={digest}")
PY
}

verify_export() {
  "$EVAL_PYTHON" - "$P03_EXPORT_DIR/export_manifest.json" "$P03_CHECKPOINT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, checkpoint = map(Path, sys.argv[1:])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if payload.get("source_checkpoint_sha256") != digest:
    raise SystemExit("strict_export_checkpoint_sha_mismatch")
if payload.get("parameters") != 63912192:
    raise SystemExit(f"strict_export_parameter_mismatch={payload.get('parameters')}")
print(f"strict_export_gate=pass checkpoint_sha256={digest}")
PY
}

do_export() {
  [[ ! -e "$P03_EXPORT_DIR" ]] || fail "stale_export_dir_without_reusable_manifest:$P03_EXPORT_DIR"
  "$EVAL_PYTHON" "$ROOT_DIR/scripts/eval/export_minimind_base.py" \
    --checkpoint "$P03_CHECKPOINT" \
    --output-dir "$P03_EXPORT_DIR" \
    --minimind-dir "$ROOT_DIR/minimind" \
    --hidden-size 768 \
    --num-hidden-layers 8 \
    --dtype float16
  verify_export
}

verify_validation() {
  local output="$1" checkpoint="$2"
  "$LAB_PYTHON" - "$output" "$checkpoint" "$EXPECTED_DATASET_FINGERPRINT" \
    "$EXPECTED_VALIDATION_ROWS" "$EXPECTED_VALIDATION_TARGET_TOKENS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output, checkpoint = map(Path, sys.argv[1:3])
fingerprint, expected_rows, expected_tokens = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
payload = json.loads(output.read_text(encoding="utf-8"))
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
checks = {
    "status": payload.get("status") == "completed",
    "checkpoint_sha256": payload.get("checkpoint_sha256") == digest,
    "dataset_fingerprint": payload.get("dataset_fingerprint") == fingerprint,
    "dataset_rows": payload.get("dataset_rows") == expected_rows,
    "rows": payload.get("rows") == expected_rows,
    "target_tokens": payload.get("target_tokens") == expected_tokens,
    "expected_rows": payload.get("expected_rows") == expected_rows,
    "expected_target_tokens": payload.get("expected_target_tokens") == expected_tokens,
}
failed = [key for key, passed in checks.items() if not passed]
if failed:
    raise SystemExit("validation_gate_failed=" + ",".join(failed))
print(f"validation_gate=pass output={output}")
PY
}

do_validation() {
  local checkpoint="$1" output="$2"
  "$LAB_PYTHON" "$ROOT_DIR/scripts/eval/eval_pretrain_validation.py" \
    --checkpoint "$checkpoint" \
    --data "$VALIDATION_FILE" \
    --dataset-success "$SUCCESS_MARKER" \
    --output "$output" \
    --minimind-dir "$ROOT_DIR/minimind" \
    --hidden-size 768 \
    --num-hidden-layers 8 \
    --max-seq-len 768 \
    --batch-size 8 \
    --device cuda:0 \
    --dtype bfloat16 \
    --expected-rows "$EXPECTED_VALIDATION_ROWS" \
    --expected-target-tokens "$EXPECTED_VALIDATION_TARGET_TOKENS"
  verify_validation "$output" "$checkpoint"
}

do_continuations() {
  "$EVAL_PYTHON" "$ROOT_DIR/scripts/eval/eval_base_continuation.py" \
    --model "$P03_EXPORT_DIR" \
    --output "$CONTINUATIONS" \
    --device cuda:0 \
    --max-new-tokens 64
  "$LAB_PYTHON" - "$CONTINUATIONS" <<'PY'
import json
import sys
from pathlib import Path

expected = ["zh_capital", "en_capital", "arithmetic", "science", "code"]
records = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
if [item.get("id") for item in records] != expected:
    raise SystemExit("continuation_five_prompt_gate_failed")
required = {"prompt", "continuation", "generated_tokens", "decoding", "max_new_tokens", "chat_template", "seed", "recorded_at"}
if any(not required.issubset(item) for item in records):
    raise SystemExit("continuation_schema_gate_failed")
print("continuation_gate=pass samples=5")
PY
}

do_official() {
  mkdir -p "$EVAL_DIR"
  set +e
  HF_HOME=/data/cache/huggingface \
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  CUDA_VISIBLE_DEVICES=0 \
  "$LM_EVAL" run \
    --model hf \
    --model_args "pretrained=$P03_EXPORT_DIR,dtype=auto,trust_remote_code=True" \
    --tasks ceval-valid cmmlu arc_easy piqa openbookqa hellaswag social_iqa \
    --batch_size 16 \
    --device cuda:0 \
    --trust_remote_code \
    --seed 42 \
    --output_path "$OFFICIAL_JSON" \
    > >(tee "$OFFICIAL_LOG") 2>&1
  local rc=$?
  set -e
  [[ "$rc" -eq 0 ]] || return "$rc"
  "$LAB_PYTHON" - "$OFFICIAL_JSON" <<'PY'
import json
import sys
from pathlib import Path

tasks = ("ceval-valid", "cmmlu", "arc_easy", "piqa", "openbookqa", "hellaswag", "social_iqa")
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
results = payload.get("results")
missing = [task for task in tasks if not isinstance(results, dict) or task not in results]
if missing:
    raise SystemExit("official_tasks_missing=" + ",".join(missing))
print("official_gate=pass tasks=7")
PY
}

do_latency() {
  "$EVAL_PYTHON" "$ROOT_DIR/scripts/eval/benchmark_base_inference.py" \
    --model "$P03_EXPORT_DIR" \
    --output "$LATENCY_JSON" \
    --device cuda:0 \
    --dtype float16 \
    --max-new-tokens 32
  "$LAB_PYTHON" - "$LATENCY_JSON" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("fixed_length_eos_token_id", "missing") is not None:
    raise SystemExit("latency_eos_policy_gate_failed")
for key, output_tokens in (("ttft", 1), ("generation", 32)):
    item = payload.get(key, {})
    if item.get("warmup_runs") != 5 or item.get("measured_runs") != 20:
        raise SystemExit(f"latency_run_count_gate_failed={key}")
    if item.get("output_tokens_per_run") != output_tokens or len(item.get("runs", [])) != 20:
        raise SystemExit(f"latency_fixed_length_gate_failed={key}")
print("latency_gate=pass warmup=5 measured=20 eos_token_id=None")
PY
}

do_system_summary() {
  local started_at per_gpu_batch_size
  started_at="$("$LAB_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["started_at"])' "$P03_RUNTIME_MANIFEST")"
  per_gpu_batch_size="$("$LAB_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["per_gpu_batch_size"])' "$P03_RUNTIME_MANIFEST")"
  "$EVAL_PYTHON" "$ROOT_DIR/scripts/eval/summarize_pretrain.py" \
    --driver-log "$P03_DRIVER_LOG" \
    --gpu-log "$P03_GPU_LOG" \
    --checkpoint "$P03_CHECKPOINT" \
    --started-at "$started_at" \
    --world-size 8 \
    --per-gpu-batch-size "$per_gpu_batch_size" \
    --max-seq-len 768 \
    --gpu-sample-interval-seconds 2 \
    --output "$SYSTEM_JSON"
}

do_swanlab() {
  "$LAB_PYTHON" "$ROOT_DIR/scripts/eval/log_eval_to_swanlab.py" \
    --validation "$P03_VALIDATION" \
    --official "$OFFICIAL_JSON" \
    --continuation "$CONTINUATIONS" \
    --latency "$LATENCY_JSON" \
    --system "$SYSTEM_JSON" \
    --project MiniMind-Lab \
    --experiment-name P03-Eval-V1-1B28-64M-Seq768 \
    --group P03 \
    --output "$SWANLAB_RECEIPT"
}

[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fail "DRY_RUN_must_be_explicitly_0_or_1"
[[ -x "$LAB_PYTHON" ]] || fail "lab_python_not_executable:$LAB_PYTHON"
[[ -x "$EVAL_PYTHON" ]] || fail "eval_python_not_executable:$EVAL_PYTHON"
[[ -x "$LM_EVAL" ]] || fail "lm_eval_not_executable:$LM_EVAL"
for script in \
  export_minimind_base.py eval_pretrain_validation.py eval_base_continuation.py \
  benchmark_base_inference.py summarize_pretrain.py log_eval_to_swanlab.py; do
  require_file "$ROOT_DIR/scripts/eval/$script"
done

if [[ "$DRY_RUN" == "0" ]]; then
  for input in \
    "$P01_CHECKPOINT" "$P02_CHECKPOINT" "$P03_CHECKPOINT" "$P03_EXIT_STATUS" \
    "$P03_RUNTIME_MANIFEST" "$P03_DRIVER_LOG" "$P03_GPU_LOG" \
    "$VALIDATION_FILE" "$SUCCESS_MARKER"; do
    require_file "$input"
  done
  verify_formal_training_exit
  mkdir -p "$EVAL_DIR" "$STEP_DIR"
fi

run_step strict-export 2 4 \
  "$P03_CHECKPOINT" "$ROOT_DIR/scripts/eval/export_minimind_base.py" \
  "$P03_EXPORT_DIR/export_manifest.json" "$P03_EXPORT_DIR/model.safetensors" \
  "$P03_EXPORT_DIR/config.json" "$P03_EXPORT_DIR/tokenizer.json" \
  -- do_export

run_step validation-p01-exact 4 1 \
  "$P01_CHECKPOINT" "$VALIDATION_FILE" "$SUCCESS_MARKER" \
  "$ROOT_DIR/scripts/eval/eval_pretrain_validation.py" "$P01_VALIDATION" \
  -- do_validation "$P01_CHECKPOINT" "$P01_VALIDATION"

run_step validation-p02-exact 4 1 \
  "$P02_CHECKPOINT" "$VALIDATION_FILE" "$SUCCESS_MARKER" \
  "$ROOT_DIR/scripts/eval/eval_pretrain_validation.py" "$P02_VALIDATION" \
  -- do_validation "$P02_CHECKPOINT" "$P02_VALIDATION"

run_step validation-p03-exact 4 1 \
  "$P03_CHECKPOINT" "$VALIDATION_FILE" "$SUCCESS_MARKER" \
  "$ROOT_DIR/scripts/eval/eval_pretrain_validation.py" "$P03_VALIDATION" \
  -- do_validation "$P03_CHECKPOINT" "$P03_VALIDATION"

run_step continuations-p03-5 4 1 \
  "$P03_EXPORT_DIR/export_manifest.json" "$P03_EXPORT_DIR/model.safetensors" \
  "$P03_EXPORT_DIR/tokenizer.json" "$ROOT_DIR/scripts/eval/eval_base_continuation.py" \
  "$CONTINUATIONS" -- do_continuations

run_step official-seven-p03 4 2 \
  "$P03_EXPORT_DIR/export_manifest.json" "$P03_EXPORT_DIR/model.safetensors" \
  "$P03_EXPORT_DIR/config.json" "$LM_EVAL" \
  "$OFFICIAL_JSON" "$OFFICIAL_LOG" -- do_official

run_step latency-p03 4 1 \
  "$P03_EXPORT_DIR/export_manifest.json" "$P03_EXPORT_DIR/model.safetensors" \
  "$P03_EXPORT_DIR/tokenizer.json" "$ROOT_DIR/scripts/eval/benchmark_base_inference.py" \
  "$LATENCY_JSON" -- do_latency

run_step system-p03 5 1 \
  "$P03_DRIVER_LOG" "$P03_GPU_LOG" "$P03_CHECKPOINT" "$P03_RUNTIME_MANIFEST" \
  "$ROOT_DIR/scripts/eval/summarize_pretrain.py" "$SYSTEM_JSON" -- do_system_summary

run_step swanlab-p03-eval 6 1 \
  "$P03_VALIDATION" "$OFFICIAL_JSON" "$CONTINUATIONS" "$LATENCY_JSON" \
  "$SYSTEM_JSON" "$ROOT_DIR/scripts/eval/log_eval_to_swanlab.py" \
  "$SWANLAB_RECEIPT" -- do_swanlab

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'evaluation_pipeline=plan_validated experiment=%s\n' "$EXPERIMENT_ID"
else
  printf 'evaluation_pipeline=completed experiment=%s\n' "$EXPERIMENT_ID"
fi
