#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$REPO_DIR/.venv}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
MODEL_DIR="${DA3_MODEL_DIR:-depth-anything/DA3METRIC-LARGE}"
PROCESS_RES="${PROCESS_RES:-504}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "No environment at $VENV. Run ./setup.sh first." >&2
  exit 1
fi

# Apple Silicon: run ops MPS lacks on the CPU instead of failing.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

exec "$VENV/bin/python" "$REPO_DIR/demo.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  --model-dir "${MODEL_DIR}" \
  --process-res "${PROCESS_RES}" \
  --depth-model "${DEPTH_MODEL:-da3}"
