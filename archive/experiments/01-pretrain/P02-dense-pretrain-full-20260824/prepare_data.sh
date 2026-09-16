#!/usr/bin/env bash
set -Eeuo pipefail

REVISION="312afb4f76391145c6902f765bb51691c09a12f5"
TARGET_DIR="${TARGET_DIR:-/data/datasets/minimind/$REVISION}"
TARGET_FILE="$TARGET_DIR/pretrain_t2t.jsonl"
EXPECTED_SIZE="8275074893"
EXPECTED_SHA256="31efc9a6fa7430769c0e78cde1c8ec0273ac7bbad20614c0ee58bccef327cc9d"
EXPECTED_LINES="8468827"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/data/cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "$TARGET_DIR"
if [[ ! -f "$TARGET_FILE" ]]; then
  /data/venvs/minimind-lab/bin/huggingface-cli download \
    jingyaogong/minimind_dataset pretrain_t2t.jsonl \
    --repo-type dataset \
    --revision "$REVISION" \
    --local-dir "$TARGET_DIR" \
    --quiet
fi

actual_size="$(stat -c %s "$TARGET_FILE")"
actual_sha256="$(sha256sum "$TARGET_FILE" | awk '{print $1}')"
actual_lines="$(wc -l < "$TARGET_FILE" | tr -d ' ')"

[[ "$actual_size" == "$EXPECTED_SIZE" ]]
[[ "$actual_sha256" == "$EXPECTED_SHA256" ]]
[[ "$actual_lines" == "$EXPECTED_LINES" ]]
printf 'verified=%s size=%s lines=%s sha256=%s\n' "$TARGET_FILE" "$actual_size" "$actual_lines" "$actual_sha256"
