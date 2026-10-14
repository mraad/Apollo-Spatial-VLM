#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VLM_VENV:-$REPO_DIR/.venv-vlm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VLLM_VERSION="${VLLM_VERSION:-0.31.0}"
TORCH_BACKEND="${TORCH_BACKEND:-auto}"
ENV_FILE="${ENV_FILE:-$REPO_DIR/.env.vlm}"

if ! command -v uv >/dev/null; then
  echo "uv is not installed: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if [ "$(uname -s)" != "Linux" ]; then
  echo "vLLM needs Linux with an NVIDIA GPU. On macOS, serve the VLM with Ollama instead (see README)." >&2
  exit 1
fi

# Separate venv: vLLM pins its own torch, which can differ from the demo's.
uv venv --allow-existing --python "$PYTHON_VERSION" "$VENV"
uv pip install --python "$VENV" --torch-backend "$TORCH_BACKEND" "vllm==${VLLM_VERSION}"

if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" <<'EOF'
QWEN_URL=http://127.0.0.1:8012/v1
GPU_UTIL=0.7
MAX_MODEL_LEN=16384
VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
EOF
fi

cat <<EOF

VLM environment ready: $VENV
Config file: $ENV_FILE

Start the server with:
  ./run-vlm.sh
EOF
