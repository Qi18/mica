#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
D02_COMMAND="$ROOT_DIR/experiments/00-preparation/D02-sft-data-v1-20260901/command.sh"
RUN_MODE="${RUN_MODE:-validate}"

test -f "$D02_COMMAND" || {
  printf 'error=missing_D02_command:%s\n' "$D02_COMMAND" >&2
  exit 66
}

case "$RUN_MODE" in
  validate)
    RUN_MODE=resolve bash "$D02_COMMAND"
    RUN_MODE=validate bash "$D02_COMMAND"
    ;;
  materialize)
    RUN_MODE=materialize bash "$D02_COMMAND"
    ;;
  *)
    printf 'error=unsupported_RUN_MODE:%s allowed=validate,materialize\n' "$RUN_MODE" >&2
    exit 2
    ;;
esac
