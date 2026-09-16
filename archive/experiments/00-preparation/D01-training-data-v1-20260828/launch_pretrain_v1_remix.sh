#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="/data/projects/minimind-lab-data-v1"
readonly DATA_ROOT="/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28"
readonly WORK_ROOT="${DATA_ROOT}/work"
readonly FINAL_ROOT="${DATA_ROOT}/final-remix-v1"
readonly EXP_ROOT="${REPO_ROOT}/experiments/00-preparation/D01-training-data-v1-20260828"
readonly LAUNCHER="$(readlink -f "${BASH_SOURCE[0]}")"
readonly RUNS_ROOT="${EXP_ROOT}/runs"
readonly PYTHON_BIN="/data/venvs/minimind-lab/bin/python"
readonly TIMEOUT_BIN="/usr/bin/timeout"
readonly AUDIT_TIMEOUT_DURATION="6h"
readonly AUDIT_TIMEOUT_SECONDS=21600
readonly AUDIT_KILL_AFTER_DURATION="30s"
readonly AUDIT_KILL_AFTER_SECONDS=30
readonly AUDIT_TIMEOUT_SIGNAL="TERM"

readonly PRETRAIN_CONFIG="${REPO_ROOT}/configs/data/pretrain/pretrain_v1.yaml"
readonly SOURCES_CONFIG="${REPO_ROOT}/configs/data/pretrain/pretrain_shards_v1.yaml"
readonly EVAL_CONFIG="${REPO_ROOT}/configs/data/pretrain/contamination_eval_v2.yaml"
readonly TOKENIZER_ROOT="${REPO_ROOT}/minimind/model"
readonly TOKENIZER_JSON="${TOKENIZER_ROOT}/tokenizer.json"
readonly TOKENIZER_CONFIG="${TOKENIZER_ROOT}/tokenizer_config.json"
readonly CANDIDATE_MANIFEST="${WORK_ROOT}/candidates/manifest.json"
readonly BUILDER="${REPO_ROOT}/scripts/data/pretrain/build_pretrain_v1.py"
readonly REMIXER="${REPO_ROOT}/scripts/data/pretrain/remix_pretrain_v1.py"
readonly AUDITOR="${REPO_ROOT}/scripts/data/pretrain/audit_pretrain_v1.py"
readonly LOADER_HOOK="${REPO_ROOT}/minimind/dataset/lm_dataset.py:PretrainDataset"
readonly BUILD_MANIFEST="${FINAL_ROOT}/manifest.json"
readonly SUCCESS_MARKER="${FINAL_ROOT}/_SUCCESS"
readonly VERIFY_REPORT="${EXP_ROOT}/pretrain_v1_remix_verification.json"
readonly AUDIT_REPORT="${EXP_ROOT}/pretrain_v1_remix_audit.json"

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/data/cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ETAG_TIMEOUT=20
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

MODE="run"
if (( $# > 1 )); then
  printf 'usage: %s [--metadata-only|--self-test-failure]\n' "$0" >&2
  exit 2
fi
case "${1:-}" in
  "") ;;
  --metadata-only) MODE="metadata-only" ;;
  --self-test-failure) MODE="self-test-failure" ;;
  -h|--help)
    printf 'usage: %s [--metadata-only|--self-test-failure]\n' "$0"
    exit 0
    ;;
  *)
    printf 'unknown option: %s\n' "$1" >&2
    exit 2
    ;;
esac
readonly MODE

utc_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

write_command() {
  local step="$1"
  shift
  printf '%s\t' "${step}"
  printf '%q ' "$@"
  printf '\n'
}

cd "${REPO_ROOT}"

readonly START_UTC="$(utc_now)"
readonly START_EPOCH="$(date -u +%s)"
readonly RUN_ID="pretrain-v1-remix-${MODE}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly RUN_DIR="${RUNS_ROOT}/${RUN_ID}"
readonly METADATA_FILE="${RUN_DIR}/metadata.json"
readonly COMMANDS_FILE="${RUN_DIR}/commands.tsv"
readonly EVENTS_FILE="${RUN_DIR}/events.tsv"
readonly STATUS_FILE="${RUN_DIR}/status.env"
readonly RUN_LOG="${RUN_DIR}/run.log"

readonly GIT_HEAD="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain=v1 --untracked-files=normal)" ]]; then
  readonly GIT_DIRTY=true
else
  readonly GIT_DIRTY=false
fi

readonly PRETRAIN_CONFIG_SHA="$(sha256_of "${PRETRAIN_CONFIG}")"
readonly SOURCES_CONFIG_SHA="$(sha256_of "${SOURCES_CONFIG}")"
readonly EVAL_CONFIG_SHA="$(sha256_of "${EVAL_CONFIG}")"
readonly TOKENIZER_JSON_SHA="$(sha256_of "${TOKENIZER_JSON}")"
readonly TOKENIZER_CONFIG_SHA="$(sha256_of "${TOKENIZER_CONFIG}")"
readonly CANDIDATE_MANIFEST_SHA="$(sha256_of "${CANDIDATE_MANIFEST}")"
readonly BUILDER_SHA="$(sha256_of "${BUILDER}")"
readonly REMIXER_SHA="$(sha256_of "${REMIXER}")"
readonly AUDITOR_SHA="$(sha256_of "${AUDITOR}")"
readonly LAUNCHER_SHA="$(sha256_of "${LAUNCHER}")"

REMIX_CMD=(
  "${PYTHON_BIN}" "${REMIXER}" remix
  --config "${PRETRAIN_CONFIG}"
  --sources-config "${SOURCES_CONFIG}"
  --eval-config "${EVAL_CONFIG}"
  --tokenizer "${TOKENIZER_ROOT}"
  --cache-dir "${HF_HOME}"
  --work-root "${WORK_ROOT}"
  --output-root "${FINAL_ROOT}"
  --sort-memory-mb 512
)
VERIFY_CMD=(
  "${PYTHON_BIN}" "${BUILDER}" verify
  --config "${PRETRAIN_CONFIG}"
  --tokenizer "${TOKENIZER_ROOT}"
  --output-root "${FINAL_ROOT}"
  --loader-hook "${LOADER_HOOK}"
  --report "${VERIFY_REPORT}"
)
AUDIT_CMD=(
  "${TIMEOUT_BIN}"
  --signal="${AUDIT_TIMEOUT_SIGNAL}"
  --kill-after="${AUDIT_KILL_AFTER_DURATION}"
  "${AUDIT_TIMEOUT_DURATION}"
  "${PYTHON_BIN}" "${AUDITOR}"
  --data-dir "${FINAL_ROOT}"
  --config "${PRETRAIN_CONFIG}"
  --eval-config "${EVAL_CONFIG}"
  --tokenizer "${TOKENIZER_ROOT}"
  --cache-dir "${HF_HOME}"
  --output "${AUDIT_REPORT}"
  --build-manifest "${BUILD_MANIFEST}"
  --verification-report "${VERIFY_REPORT}"
  --success-marker "${SUCCESS_MARKER}"
  --near-mode stratified
  --near-sample-per-shard 512
)

ACCEPTANCE_PY='import hashlib
import json
import sys
from pathlib import Path

*path_values, candidate_start_sha = sys.argv[1:]
(
    marker_path,
    report_path,
    manifest_path,
    verification_path,
    candidate_path,
    config_path,
    eval_path,
    auditor_path,
    remixer_path,
) = map(Path, path_values)
sys.path.insert(0, str(auditor_path.parent))
import audit_pretrain_v1 as auditor

def read(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root is not an object")
    return value

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

for path in (
    marker_path,
    report_path,
    manifest_path,
    verification_path,
    candidate_path,
):
    if not path.is_file():
        raise FileNotFoundError(path)
candidate_sha = sha(candidate_path)
if candidate_sha != candidate_start_sha:
    raise ValueError("candidate manifest changed since launcher startup")
report = read(report_path)
manifest = read(manifest_path)
verification = read(verification_path)
if report.get("passed") is not True:
    raise ValueError("audit report did not pass")
checks = report.get("checks")
if not isinstance(checks, dict) or not checks or not all(checks.values()):
    raise ValueError("audit checks are absent or not all true")
if verification.get("manifest_sha256") != sha(manifest_path):
    raise ValueError("verification does not bind the build manifest")
if manifest.get("candidate_manifest_sha256") != candidate_sha:
    raise ValueError("build manifest does not bind the candidate manifest")
if manifest.get("contamination_filter", {}).get("status") != "applied":
    raise ValueError("contamination filter evidence is not applied")
if manifest.get("repair_mixer", {}).get("script", {}).get("sha256") != sha(remixer_path):
    raise ValueError("build manifest does not bind the current remixer")
dataset_fingerprint = report.get("dataset_fingerprint")
if not isinstance(dataset_fingerprint, str) or len(dataset_fingerprint) != 64:
    raise ValueError("audit report dataset fingerprint is missing or malformed")
repair = manifest.get("repair_mixer")
contamination_filter = manifest.get("contamination_filter")
if not isinstance(repair, dict) or not isinstance(contamination_filter, dict):
    raise ValueError("repair evidence is absent from build manifest")
repair_script = repair.get("script")
if not isinstance(repair_script, dict):
    raise ValueError("repair script evidence is absent from build manifest")
repair_evidence = {
    "script_sha256": repair_script.get("sha256"),
    "sources_config_sha256": manifest.get("sources_config_sha256"),
    "candidate_manifest_sha256": candidate_sha,
    "candidate_roster_sha256": repair.get("candidate_roster_sha256"),
    "eval_config_sha256": contamination_filter.get("eval_config_sha256"),
    "pattern_sha256": contamination_filter.get("pattern_sha256"),
    "benchmark_snapshot_sha256": contamination_filter.get(
        "benchmark_snapshot_sha256"
    ),
    "exclusion_ledger_sha256": contamination_filter.get(
        "exclusion_ledger_sha256"
    ),
    "exclusion_snapshot_sha256": contamination_filter.get(
        "exclusion_snapshot_sha256"
    ),
}
tokenizer_sha = hashlib.sha256(
    auditor.canonical_json(manifest["tokenizer"]).encode("utf-8")
).hexdigest()
expected = {
    "status": "accepted",
    "audit_report_path": str(report_path.resolve()),
    "audit_report_sha256": sha(report_path),
    "auditor_path": str(auditor_path.resolve()),
    "auditor_version": auditor.AUDITOR_VERSION,
    "auditor_sha256": sha(auditor_path),
    "manifest_path": str(manifest_path.resolve()),
    "manifest_sha256": sha(manifest_path),
    "manifest_fingerprint": manifest.get("fingerprint"),
    "verification_path": str(verification_path.resolve()),
    "verification_sha256": sha(verification_path),
    "builder_version": manifest.get("builder_version"),
    "candidate_manifest_sha256": candidate_sha,
    "config_path": str(config_path.resolve()),
    "config_sha256": sha(config_path),
    "eval_config_path": str(eval_path.resolve()),
    "eval_config_sha256": sha(eval_path),
    "tokenizer_fingerprint_sha256": tokenizer_sha,
    "dataset_fingerprint": dataset_fingerprint,
    "repair_evidence": repair_evidence,
    "benchmark_near_mode": "stratified",
    "benchmark_near_sample_per_shard": 512,
}
auditor.validate_success_marker(marker_path, expected)
print(auditor.canonical_json({
    "audit_report_sha256": expected["audit_report_sha256"],
    "dataset_fingerprint": expected["dataset_fingerprint"],
    "manifest_sha256": expected["manifest_sha256"],
    "status": "acceptance_readback_ok",
}))'
readonly ACCEPTANCE_PY
ACCEPTANCE_CMD=(
  "${PYTHON_BIN}" -c "${ACCEPTANCE_PY}"
  "${SUCCESS_MARKER}"
  "${AUDIT_REPORT}"
  "${BUILD_MANIFEST}"
  "${VERIFY_REPORT}"
  "${CANDIDATE_MANIFEST}"
  "${PRETRAIN_CONFIG}"
  "${EVAL_CONFIG}"
  "${AUDITOR}"
  "${REMIXER}"
  "${CANDIDATE_MANIFEST_SHA}"
)

mkdir -p "${RUN_DIR}"
{
  write_command remix "${REMIX_CMD[@]}"
  write_command verify "${VERIFY_CMD[@]}"
  write_command audit "${AUDIT_CMD[@]}"
  write_command acceptance_readback "${ACCEPTANCE_CMD[@]}"
} >"${COMMANDS_FILE}"
readonly COMMANDS_SHA="$(sha256_of "${COMMANDS_FILE}")"

readonly METADATA_TMP="${METADATA_FILE}.tmp"
"${PYTHON_BIN}" -c '
import json
import sys

(
    output_path, metadata_file, run_id, mode, started_at_utc, host, git_head, git_dirty,
    launcher, launcher_sha256, python_bin, timeout_bin, audit_timeout_seconds,
    audit_kill_after_seconds, audit_timeout_signal, hf_hub_etag_timeout,
    hf_hub_download_timeout, hf_hub_disable_xet, repo_root, data_root,
    work_root, final_root, pretrain_config, pretrain_config_sha256,
    sources_config, sources_config_sha256, eval_config, eval_config_sha256,
    tokenizer_root, tokenizer_json_sha256, tokenizer_config_sha256,
    candidate_manifest, candidate_manifest_sha256, builder, builder_sha256,
    remixer, remixer_sha256, auditor, auditor_sha256, build_manifest,
    verification_report, audit_report, success_marker, commands_file,
    commands_sha256, events_file, status_file, run_log,
) = sys.argv[1:]

payload = {
    "schema_version": 1,
    "run_id": run_id,
    "mode": mode,
    "started_at_utc": started_at_utc,
    "host": host,
    "git": {"head": git_head, "dirty": git_dirty.lower() == "true"},
    "launcher": {"path": launcher, "sha256": launcher_sha256},
    "runtime": {
        "python": python_bin,
        "policy": {
            "audit": {
                "timeout_binary": timeout_bin,
                "timeout_seconds": int(audit_timeout_seconds),
                "terminate_signal": audit_timeout_signal,
                "kill_after_seconds": int(audit_kill_after_seconds),
            },
            "hf_hub": {
                "endpoint": "https://hf-mirror.com",
                "etag_timeout_seconds": int(hf_hub_etag_timeout),
                "download_timeout_seconds": int(hf_hub_download_timeout),
                "disable_xet": hf_hub_disable_xet == "1",
            },
        },
    },
    "paths": {
        "repo_root": repo_root,
        "data_root": data_root,
        "work_root": work_root,
        "final_root": final_root,
        "build_manifest": build_manifest,
        "verification_report": verification_report,
        "audit_report": audit_report,
        "success_marker": success_marker,
    },
    "inputs": {
        "pretrain_config": {"path": pretrain_config, "sha256": pretrain_config_sha256},
        "sources_config": {"path": sources_config, "sha256": sources_config_sha256},
        "eval_config": {"path": eval_config, "sha256": eval_config_sha256},
        "tokenizer": {
            "path": tokenizer_root,
            "tokenizer_json_sha256": tokenizer_json_sha256,
            "tokenizer_config_sha256": tokenizer_config_sha256,
        },
        "candidate_manifest": {"path": candidate_manifest, "sha256": candidate_manifest_sha256},
        "builder": {"path": builder, "sha256": builder_sha256},
        "remixer": {"path": remixer, "sha256": remixer_sha256},
        "auditor": {"path": auditor, "sha256": auditor_sha256},
    },
    "commands": {"path": commands_file, "sha256": commands_sha256},
    "run_artifacts": {
        "metadata": {"path": metadata_file},
        "commands": {"path": commands_file, "sha256": commands_sha256},
        "events": {"path": events_file},
        "status": {"path": status_file},
        "log": {"path": run_log},
    },
    "security": {"environment_recorded": False, "credentials_recorded": False},
}
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")
' \
  "${METADATA_TMP}" "${METADATA_FILE}" "${RUN_ID}" "${MODE}" \
  "${START_UTC}" "$(hostname)" "${GIT_HEAD}" "${GIT_DIRTY}" \
  "${LAUNCHER}" "${LAUNCHER_SHA}" "${PYTHON_BIN}" \
  "${TIMEOUT_BIN}" "${AUDIT_TIMEOUT_SECONDS}" "${AUDIT_KILL_AFTER_SECONDS}" \
  "${AUDIT_TIMEOUT_SIGNAL}" "${HF_HUB_ETAG_TIMEOUT}" \
  "${HF_HUB_DOWNLOAD_TIMEOUT}" "${HF_HUB_DISABLE_XET}" "${REPO_ROOT}" \
  "${DATA_ROOT}" "${WORK_ROOT}" "${FINAL_ROOT}" "${PRETRAIN_CONFIG}" \
  "${PRETRAIN_CONFIG_SHA}" "${SOURCES_CONFIG}" "${SOURCES_CONFIG_SHA}" \
  "${EVAL_CONFIG}" "${EVAL_CONFIG_SHA}" "${TOKENIZER_ROOT}" \
  "${TOKENIZER_JSON_SHA}" "${TOKENIZER_CONFIG_SHA}" \
  "${CANDIDATE_MANIFEST}" "${CANDIDATE_MANIFEST_SHA}" "${BUILDER}" \
  "${BUILDER_SHA}" "${REMIXER}" "${REMIXER_SHA}" "${AUDITOR}" \
  "${AUDITOR_SHA}" "${BUILD_MANIFEST}" "${VERIFY_REPORT}" "${AUDIT_REPORT}" \
  "${SUCCESS_MARKER}" "${COMMANDS_FILE}" "${COMMANDS_SHA}" \
  "${EVENTS_FILE}" "${STATUS_FILE}" "${RUN_LOG}"
mv "${METADATA_TMP}" "${METADATA_FILE}"

readonly METADATA_SHA="$(sha256_of "${METADATA_FILE}")"
CURRENT_STEP="initializing"

record_event() {
  local step="$1"
  local event="$2"
  local exit_code="${3:-}"
  printf '%s\t%s\t%s\t%s\n' \
    "$(utc_now)" "${step}" "${event}" "${exit_code}" >>"${EVENTS_FILE}"
}

write_status() {
  local status="$1"
  local exit_code="$2"
  local ended_at="$3"
  local duration_seconds="$4"
  local temporary="${STATUS_FILE}.tmp"
  {
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'status=%s\n' "${status}"
    printf 'exit_code=%s\n' "${exit_code}"
    printf 'current_step=%s\n' "${CURRENT_STEP}"
    printf 'started_at_utc=%s\n' "${START_UTC}"
    printf 'ended_at_utc=%s\n' "${ended_at}"
    printf 'duration_seconds=%s\n' "${duration_seconds}"
    printf 'metadata_sha256=%s\n' "${METADATA_SHA}"
    printf 'commands_sha256=%s\n' "${COMMANDS_SHA}"
  } >"${temporary}"
  mv "${temporary}" "${STATUS_FILE}"
}

on_exit() {
  local exit_code="$1"
  trap - EXIT
  set +e
  local ended_at end_epoch duration outcome
  outcome=failed
  ended_at="$(utc_now)"
  end_epoch="$(date -u +%s)"
  duration="$((end_epoch - START_EPOCH))"
  if [[ "${exit_code}" -eq 0 && "${CURRENT_STEP}" == "complete" ]]; then
    outcome=success
  fi
  record_event "${CURRENT_STEP}" run_exit "${exit_code}"
  write_status "${outcome}" "${exit_code}" "${ended_at}" "${duration}"
  printf 'run_id=%s status=%s exit_code=%s metadata=%s\n' \
    "${RUN_ID}" "${outcome}" "${exit_code}" "${METADATA_FILE}"
  exit "${exit_code}"
}

run_step() {
  local step="$1"
  shift
  local exit_code
  CURRENT_STEP="${step}"
  record_event "${step}" start
  if "$@"; then
    record_event "${step}" success 0
    return 0
  else
    exit_code="$?"
    record_event "${step}" failed "${exit_code}"
    return "${exit_code}"
  fi
}

trap 'on_exit $?' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

: >"${EVENTS_FILE}"
write_status running "" "" 0
exec > >(tee -a "${RUN_LOG}") 2>&1

printf 'run_id=%s started_at_utc=%s metadata=%s\n' \
  "${RUN_ID}" "${START_UTC}" "${METADATA_FILE}"
record_event initializing run_started 0

case "${MODE}" in
  metadata-only)
    CURRENT_STEP=complete
    record_event complete metadata_only_complete 0
    exit 0
    ;;
  self-test-failure)
    CURRENT_STEP=self_test_pre_remix
    record_event "${CURRENT_STEP}" controlled_failure 97
    exit 97
    ;;
esac

run_step remix "${REMIX_CMD[@]}"
run_step verify "${VERIFY_CMD[@]}"
run_step audit "${AUDIT_CMD[@]}"
run_step acceptance_readback "${ACCEPTANCE_CMD[@]}"

CURRENT_STEP=complete
record_event complete pipeline_complete 0
