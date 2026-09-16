#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
export MINIMIND_PYTHON=/data/venvs/minimind-lab/bin/python
export SWANLAB_BIN=/data/venvs/minimind-lab/bin/swanlab

"$ROOT_DIR/scripts/launch/preflight_l20.py"
"$MINIMIND_PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  "$ROOT_DIR/scripts/launch/nccl_smoke.py"
