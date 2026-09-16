#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXPERIMENT_ID="P03-dense-pretrain-v1-1b28-20260901"
EXP_DIR="$ROOT_DIR/experiments/01-pretrain/$EXPERIMENT_ID"
CONFIG_PATH="$EXP_DIR/config.json"
COMMAND_PATH="$EXP_DIR/command.sh"
TRAINER_PATH="$ROOT_DIR/minimind/trainer/train_pretrain_with_validation.py"
DATA_ROOT="/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/final-remix-v1"
TRAIN_GLOB="$DATA_ROOT/train-*-of-00040.jsonl"
VALIDATION_GLOB="$DATA_ROOT/validation-*-of-00001.jsonl"
SUCCESS_MARKER="$DATA_ROOT/_SUCCESS"
DATA_MANIFEST="$DATA_ROOT/manifest.json"
EXPECTED_SUCCESS_SHA256="8c9c728ed063b214ebb74e905e0f6ccc4a510c7078408b87cda5d0c990639c7c"
EXPECTED_MANIFEST_SHA256="1d14286c760e33884a5bc8d8afd1ac95e9d084f1b52b9bf2f582fc3743b694d6"
EXPECTED_DATASET_FINGERPRINT="cd018f6d0a047284f5f77d240d2583a1673c9d9a923536e9da7e4b1e4ead70bd"
PYTHON_BIN="${PYTHON_BIN:-/data/venvs/minimind-lab/bin/python}"
LAUNCH_STARTED_EPOCH_NS="$(date +%s%N)"
RUN_KIND="${RUN_KIND:-}"
FROM_RESUME="${FROM_RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-32}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
WORLD_SIZE=8
FULL_OPTIMIZER_STEPS=9038
ARTIFACT_ROOT="/data/artifacts/minimind-lab/$EXPERIMENT_ID"
MONITOR_PID=""

fail() {
  printf 'error=%s\n' "$1" >&2
  exit 1
}

cleanup() {
  if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM HUP

[[ -x "$PYTHON_BIN" ]] || fail "python_not_executable:$PYTHON_BIN"
[[ -r "$TRAINER_PATH" ]] || fail "trainer_not_readable:$TRAINER_PATH"
[[ -r "$CONFIG_PATH" ]] || fail "config_not_readable:$CONFIG_PATH"
[[ -r "$SUCCESS_MARKER" ]] || fail "accepted_marker_not_readable:$SUCCESS_MARKER"
[[ -r "$DATA_MANIFEST" ]] || fail "manifest_not_readable:$DATA_MANIFEST"
[[ "$RUN_KIND" == "probe" || "$RUN_KIND" == "formal" ]] || fail "RUN_KIND_must_be_probe_or_formal"
[[ "$FROM_RESUME" == "0" || "$FROM_RESUME" == "1" ]] || fail "FROM_RESUME_must_be_0_or_1"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fail "DRY_RUN_must_be_0_or_1"
[[ "$PER_GPU_BATCH_SIZE" =~ ^[0-9]+$ ]] || fail "PER_GPU_BATCH_SIZE_must_be_integer"
[[ "$ACCUMULATION_STEPS" =~ ^[0-9]+$ ]] || fail "ACCUMULATION_STEPS_must_be_integer"

TRAINER_SHA256="$(sha256sum "$TRAINER_PATH" | awk '{print $1}')"
PROTOCOL_SHA256="$(sha256sum "$CONFIG_PATH" | awk '{print $1}')"
COMMAND_SHA256="$(sha256sum "$COMMAND_PATH" | awk '{print $1}')"

case "$PER_GPU_BATCH_SIZE:$ACCUMULATION_STEPS" in
  32:1)
    EXPECTED_MICRO_STEPS=9038
    ;;
  16:2)
    EXPECTED_MICRO_STEPS=18075
    ;;
  8:4)
    EXPECTED_MICRO_STEPS=36149
    ;;
  *)
    fail "unsupported_batch_accum_profile:expected_32x1_16x2_or_8x4"
    ;;
esac

GLOBAL_SEQUENCE_BATCH=$((PER_GPU_BATCH_SIZE * ACCUMULATION_STEPS * WORLD_SIZE))
[[ "$GLOBAL_SEQUENCE_BATCH" -eq 256 ]] || fail "global_sequence_batch_must_equal_256"

if [[ "$RUN_KIND" == "probe" ]]; then
  MAX_STEPS="${MAX_STEPS:-100}"
  [[ "$MAX_STEPS" == "100" ]] || fail "probe_MAX_STEPS_must_equal_100"
  RUN_SLUG="probe-b${PER_GPU_BATCH_SIZE}-a${ACCUMULATION_STEPS}-step100"
  SWANLAB_RUN_NAME="P03-Pretrain-V1-1B28-Full-64M-Seq768-Probe-B${PER_GPU_BATCH_SIZE}x8-A${ACCUMULATION_STEPS}-S100"
  SAVE_INTERVAL=50
  EVAL_INTERVAL=0
else
  MAX_STEPS="${MAX_STEPS:-0}"
  [[ "$MAX_STEPS" == "0" ]] || fail "formal_MAX_STEPS_must_equal_0"
  RUN_SLUG="formal-b${PER_GPU_BATCH_SIZE}-a${ACCUMULATION_STEPS}-epoch1"
  SWANLAB_RUN_NAME="P03-Pretrain-V1-1B28-Full-64M-Seq768-B${PER_GPU_BATCH_SIZE}x8-A${ACCUMULATION_STEPS}"
  SAVE_INTERVAL=250
  EVAL_INTERVAL=1000
fi

RUN_DIR="$ARTIFACT_ROOT/$RUN_SLUG"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
RESUME_DIR="$RUN_DIR/resume"
METRICS_PATH="$RUN_DIR/metrics.jsonl"
SAVE_WEIGHT="p03_${RUN_KIND}_b${PER_GPU_BATCH_SIZE}_a${ACCUMULATION_STEPS}"
BEST_WEIGHT="${SAVE_WEIGHT}_best_val"
RESUME_PATH="$RESUME_DIR/${SAVE_WEIGHT}_768_resume.pth"
PROBE_DIR="$ARTIFACT_ROOT/probe-b${PER_GPU_BATCH_SIZE}-a${ACCUMULATION_STEPS}-step100"

actual_success_sha256="$(sha256sum "$SUCCESS_MARKER" | awk '{print $1}')"
actual_manifest_sha256="$(sha256sum "$DATA_MANIFEST" | awk '{print $1}')"
[[ "$actual_success_sha256" == "$EXPECTED_SUCCESS_SHA256" ]] || fail "accepted_marker_sha256_mismatch:$actual_success_sha256"
[[ "$actual_manifest_sha256" == "$EXPECTED_MANIFEST_SHA256" ]] || fail "manifest_sha256_mismatch:$actual_manifest_sha256"

"$PYTHON_BIN" - "$SUCCESS_MARKER" "$DATA_MANIFEST" "$DATA_ROOT" "$EXPECTED_DATASET_FINGERPRINT" <<'PY'
import glob
import hashlib
import json
import sys
from pathlib import Path

success_path, manifest_path, data_root, expected_fingerprint = sys.argv[1:]
success = json.loads(Path(success_path).read_text(encoding="utf-8"))
manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
errors = []

def require(condition, message):
    if not condition:
        errors.append(message)

require(success.get("status") == "accepted", "success.status")
require(success.get("dataset_fingerprint") == expected_fingerprint, "dataset_fingerprint")
require(success.get("manifest", {}).get("sha256") == "1d14286c760e33884a5bc8d8afd1ac95e9d084f1b52b9bf2f582fc3743b694d6", "success.manifest.sha256")
require(manifest.get("sequence_length") == 768, "manifest.sequence_length")
require(manifest.get("train", {}).get("stats", {}).get("rows") == 2313483, "train.rows")
require(manifest.get("train", {}).get("stats", {}).get("loss_target_tokens") == 1280000000, "train.loss_target_tokens")
require(manifest.get("validation", {}).get("stats", {}).get("rows") == 11525, "validation.rows")
require(manifest.get("validation", {}).get("stats", {}).get("loss_target_tokens") == 6400000, "validation.loss_target_tokens")
train_files = sorted(glob.glob(str(Path(data_root) / "train-*-of-00040.jsonl")))
validation_files = sorted(glob.glob(str(Path(data_root) / "validation-*-of-00001.jsonl")))
require(len(train_files) == 40, f"train.shards={len(train_files)}")
require(len(validation_files) == 1, f"validation.shards={len(validation_files)}")
manifest_files = []
total_size_bytes = 0
for split, actual_files in (("train", train_files), ("validation", validation_files)):
    shards = manifest.get(split, {}).get("shards", [])
    expected_names = [entry.get("file") for entry in shards]
    actual_names = [Path(path).name for path in actual_files]
    require(len(expected_names) == (40 if split == "train" else 1), f"{split}.manifest_shards")
    require(len(set(expected_names)) == len(expected_names), f"{split}.duplicate_manifest_file")
    require(sorted(expected_names) == sorted(actual_names), f"{split}.shard_set")
    for entry in shards:
        shard_path = Path(data_root) / str(entry.get("file"))
        manifest_files.append(shard_path)
        if not shard_path.is_file():
            errors.append(f"missing:{shard_path.name}")
            continue
        actual_size = shard_path.stat().st_size
        total_size_bytes += actual_size
        require(actual_size == entry.get("size_bytes"), f"size:{shard_path.name}")
        digest = hashlib.sha256()
        with shard_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        require(digest.hexdigest() == entry.get("sha256"), f"sha256:{shard_path.name}")
require(len(manifest_files) == 41, f"manifest_files={len(manifest_files)}")
require(total_size_bytes == 3804930498, f"total_size_bytes={total_size_bytes}")
if errors:
    raise SystemExit("dataset_gate_failed=" + ",".join(errors))
print(json.dumps({
    "dataset_gate": "pass",
    "train_rows": 2313483,
    "train_loss_target_tokens": 1280000000,
    "validation_rows": 11525,
    "validation_loss_target_tokens": 6400000,
    "train_shards": len(train_files),
    "validation_shards": len(validation_files),
    "verified_shards": len(manifest_files),
    "verified_size_bytes": total_size_bytes,
}, sort_keys=True))
PY

if [[ "$RUN_KIND" == "formal" ]]; then
  "$PYTHON_BIN" - "$PROBE_DIR" "$TRAINER_SHA256" "$PROTOCOL_SHA256" "$COMMAND_SHA256" "$EXPECTED_DATASET_FINGERPRINT" "$PER_GPU_BATCH_SIZE" "$ACCUMULATION_STEPS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

probe_dir, trainer_sha, protocol_sha, command_sha, fingerprint, batch, accum = sys.argv[1:]
root = Path(probe_dir)

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

metrics_path = root / "metrics.jsonl"
if not metrics_path.is_file():
    raise SystemExit(f"formal_probe_gate_failed=metrics_missing:{metrics_path}")
records = [
    json.loads(line)
    for line in metrics_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
completed = [record for record in records if record.get("event") == "completed"]
final_validation = [
    record
    for record in records
    if record.get("event") == "validation" and record.get("phase") == "final"
]
schedule = [record for record in records if record.get("event") == "schedule"]
if not completed or not final_validation or not schedule:
    raise SystemExit("formal_probe_gate_failed=incomplete_probe_metrics")
completed = completed[-1]
final_validation = final_validation[-1]
schedule = schedule[-1]
checks = {
    "completed_status": completed.get("status") == "max_steps_reached",
    "completed_optimizer_step": completed.get("optimizer_step") == 100,
    "completed_total_schedule": completed.get("resolved_total_optimizer_steps") == 9038,
    "final_validation_rows": final_validation.get("validation_rows") == 11525,
    "final_validation_tokens": final_validation.get("validation_tokens") == 6400000,
    "schedule_total": schedule.get("resolved_total_optimizer_steps") == 9038,
    "schedule_target": schedule.get("resolved_target_optimizer_steps") == 100,
}
failed = [key for key, passed in checks.items() if not passed]
if failed:
    raise SystemExit("formal_probe_gate_failed=" + ",".join(failed))

accepted_attempt = None
for attempt in sorted((root / "attempts").glob("*"), reverse=True):
    status_path = attempt / "exit-status.json"
    runtime_path = attempt / "runtime-manifest.json"
    if not status_path.is_file() or not runtime_path.is_file():
        continue
    try:
        status = load(status_path)
        runtime = load(runtime_path)
        checkpoint = Path(status.get("checkpoint", ""))
        attempt_ok = (
            status.get("exit_code") == 0
            and status.get("checkpoint_exists") is True
            and checkpoint.is_file()
            and status.get("checkpoint_sha256") == sha(checkpoint)
            and float(status.get("launcher_wall_seconds", 0)) > 0
            and runtime.get("run_kind") == "probe"
            and runtime.get("trainer_sha256") == trainer_sha
            and runtime.get("protocol_sha256") == protocol_sha
            and runtime.get("command_sha256") == command_sha
            and runtime.get("dataset_fingerprint") == fingerprint
            and runtime.get("per_gpu_batch_size") == int(batch)
            and runtime.get("accumulation_steps") == int(accum)
            and runtime.get("max_steps") == 100
            and runtime.get("resolved_total_optimizer_steps") == 9038
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        attempt_ok = False
    if attempt_ok:
        accepted_attempt = attempt.name
        break
if accepted_attempt is None:
    raise SystemExit("formal_probe_gate_failed=no_same_protocol_exit0_attempt")
print(json.dumps({"formal_probe_gate": "pass", "probe_attempt": accepted_attempt}, sort_keys=True))
PY
fi

if [[ "$FROM_RESUME" == "1" ]]; then
  [[ -r "$RESUME_PATH" ]] || fail "resume_checkpoint_not_found:$RESUME_PATH"
elif [[ -e "$RUN_DIR" ]]; then
  fail "stale_run_dir_exists:$RUN_DIR"
fi

export HF_HOME=/data/cache/huggingface
export HF_DATASETS_CACHE=/data/cache/huggingface/datasets
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export MINIMIND_PYTHON="$PYTHON_BIN"
export REQUIRE_SWANLAB=1
export EXPECTED_GPU_COUNT="$WORLD_SIZE"

GUARD_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
  GUARD_ARGS+=(--dry-run)
else
  "$PYTHON_BIN" - "$TRAIN_GLOB" "$VALIDATION_GLOB" <<'PY'
import sys
from datasets import load_dataset

train_path, validation_path = sys.argv[1:]
train = load_dataset("json", data_files=train_path, split="train")
validation = load_dataset("json", data_files=validation_path, split="train")
if len(train) != 2313483:
    raise SystemExit(f"train_cache_row_mismatch={len(train)}")
if len(validation) != 11525:
    raise SystemExit(f"validation_cache_row_mismatch={len(validation)}")
print(f"cache_gate=pass train_rows={len(train)} validation_rows={len(validation)}")
PY
fi

TRAIN_COMMAND=(
  "$PYTHON_BIN" -m torch.distributed.run
  --nproc_per_node="$WORLD_SIZE"
  train_pretrain_with_validation.py
  --epochs 1
  --batch_size "$PER_GPU_BATCH_SIZE"
  --accumulation_steps "$ACCUMULATION_STEPS"
  --max_seq_len 768
  --hidden_size 768
  --num_hidden_layers 8
  --use_moe 0
  --dtype bfloat16
  --learning_rate 5e-4
  --grad_clip 1.0
  --num_workers 8
  --log_interval 10
  --save_interval "$SAVE_INTERVAL"
  --eval_interval "$EVAL_INTERVAL"
  --max_steps "$MAX_STEPS"
  --data_path "$TRAIN_GLOB"
  --validation_path "$VALIDATION_GLOB"
  --validation_batch_size 64
  --expected_validation_rows 11525
  --expected_validation_tokens 6400000
  --save_dir "$CHECKPOINT_DIR"
  --resume_dir "$RESUME_DIR"
  --save_weight "$SAVE_WEIGHT"
  --best_weight "$BEST_WEIGHT"
  --from_weight none
  --from_resume "$FROM_RESUME"
  --metrics_path "$METRICS_PATH"
  --experiment_id "$EXPERIMENT_ID"
  --dataset_fingerprint "$EXPECTED_DATASET_FINGERPRINT"
  --trainer_sha256 "$TRAINER_SHA256"
  --protocol_sha256 "$PROTOCOL_SHA256"
  --use_swanlab
  --swanlab_project MiniMind-Lab
  --swanlab_group Pretrain
  --swanlab_tags "pretrain,P03,full,64M,seq768,$RUN_KIND"
  --swanlab_run_name "$SWANLAB_RUN_NAME"
)

cd "$ROOT_DIR/minimind/trainer"

if [[ "$DRY_RUN" == "1" ]]; then
  "$ROOT_DIR/scripts/launch/run_guarded.py" "${GUARD_ARGS[@]}" -- "${TRAIN_COMMAND[@]}"
  exit 0
fi

ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ATTEMPT_DIR="$RUN_DIR/attempts/$ATTEMPT_ID"
DRIVER_LOG="$ATTEMPT_DIR/driver.log"
HARDWARE_LOG="$ATTEMPT_DIR/nvidia-smi.csv"
RUNTIME_MANIFEST="$ATTEMPT_DIR/runtime-manifest.json"
EXIT_STATUS="$ATTEMPT_DIR/exit-status.json"
mkdir -p "$CHECKPOINT_DIR" "$RESUME_DIR" "$ATTEMPT_DIR"
"$PYTHON_BIN" - "$RUNTIME_MANIFEST" "$ROOT_DIR" "$CONFIG_PATH" "$PER_GPU_BATCH_SIZE" "$ACCUMULATION_STEPS" "$MAX_STEPS" "$EXPECTED_MICRO_STEPS" "$SWANLAB_RUN_NAME" "$RUN_DIR" "$ATTEMPT_ID" "$TRAINER_SHA256" "$PROTOCOL_SHA256" "$COMMAND_SHA256" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

output, root, config, batch, accum, max_steps, micro_steps, run_name, run_dir, attempt_id, trainer_sha, protocol_sha, command_sha = sys.argv[1:]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

payload = {
    "experiment_id": "P03-dense-pretrain-v1-1b28-20260901",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "lab_commit": subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip(),
    "git_status_porcelain": subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True).splitlines(),
    "dataset_fingerprint": "cd018f6d0a047284f5f77d240d2583a1673c9d9a923536e9da7e4b1e4ead70bd",
    "manifest_sha256": "1d14286c760e33884a5bc8d8afd1ac95e9d084f1b52b9bf2f582fc3743b694d6",
    "trainer_sha256": trainer_sha,
    "protocol_sha256": protocol_sha,
    "config_sha256": sha(config),
    "command_sha256": command_sha,
    "run_kind": "probe" if max_steps == "100" else "formal",
    "run_dir": run_dir,
    "attempt_id": attempt_id,
    "swanlab_project": "MiniMind-Lab",
    "swanlab_run_name": run_name,
    "world_size": 8,
    "per_gpu_batch_size": int(batch),
    "accumulation_steps": int(accum),
    "global_sequence_batch": int(batch) * int(accum) * 8,
    "expected_micro_steps_per_epoch": int(micro_steps),
    "resolved_total_optimizer_steps": 9038,
    "max_steps": int(max_steps),
    "max_steps_is_exit_bound_only": True,
}
path = Path(output)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY

nvidia-smi   --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu   --format=csv,noheader,nounits   --loop=2 > "$HARDWARE_LOG" &
MONITOR_PID="$!"

set +e
"$ROOT_DIR/scripts/launch/run_guarded.py" "${GUARD_ARGS[@]}" -- "${TRAIN_COMMAND[@]}" 2>&1 | tee "$DRIVER_LOG"
RUN_RC=${PIPESTATUS[0]}
set -e

cleanup
MONITOR_PID=""
"$PYTHON_BIN" - "$EXIT_STATUS" "$RUN_RC" "$CHECKPOINT_DIR/${SAVE_WEIGHT}_768.pth" "$LAUNCH_STARTED_EPOCH_NS" "$ATTEMPT_ID" <<'PY'
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

output, return_code, checkpoint, launch_started_ns, attempt_id = sys.argv[1:]
checkpoint_path = Path(checkpoint)
payload = {
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "attempt_id": attempt_id,
    "exit_code": int(return_code),
    "launcher_wall_seconds": (time.time_ns() - int(launch_started_ns)) / 1e9,
    "checkpoint": str(checkpoint_path),
    "checkpoint_exists": checkpoint_path.is_file(),
    "checkpoint_sha256": (
        hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if checkpoint_path.is_file()
        else None
    ),
}
path = Path(output)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY

exit "$RUN_RC"
