#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
printf 'Replace this placeholder with the reviewed training command.\n' >&2
printf 'Use %s/scripts/launch/run_guarded.py -- <command>.\n' "$ROOT_DIR" >&2
exit 2
