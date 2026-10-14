#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$REPO_DIR/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
# "auto" picks the torch build that matches the installed NVIDIA driver (or CPU/MPS); e.g. TORCH_BACKEND=cu128 to force one.
TORCH_BACKEND="${TORCH_BACKEND:-auto}"

if ! command -v uv >/dev/null; then
  echo "uv is not installed: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if [ ! -f "$REPO_DIR/Depth-Anything-3/pyproject.toml" ]; then
  git -C "$REPO_DIR" submodule update --init --recursive
fi

uv venv --allow-existing --python "$PYTHON_VERSION" "$VENV"
uv pip install --python "$VENV" --torch-backend "$TORCH_BACKEND" \
  torch \
  torchvision \
  fastapi \
  "uvicorn[standard]" \
  python-multipart \
  pillow \
  numpy \
  opencv-python \
  huggingface_hub \
  einops \
  omegaconf \
  safetensors \
  imageio \
  addict \
  tqdm \
  evo \
  "moviepy==1.0.3" \
  plyfile \
  trimesh \
  requests

# Depth-Anything-3 imports pycolmap at load time; there is no wheel for Linux aarch64 (e.g. Jetson).
if ! uv pip install --python "$VENV" pycolmap; then
  echo "No pycolmap wheel for this platform. Build COLMAP from source and run" >&2
  echo "  CMAKE_PREFIX_PATH=<colmap install> uv pip install --python $VENV <colmap repo>" >&2
  echo "then re-run ./setup.sh (see README)." >&2
  exit 1
fi

uv pip install --python "$VENV" --no-deps -e "$REPO_DIR/Depth-Anything-3"

# Optional MoGe-3 depth models (Linux + NVIDIA only: FlexGEMM needs Triton). Pinned to the commit tested on the RTX PRO 6000 server.
if [ "${WITH_MOGE:-0}" = "1" ]; then
  uv pip install --python "$VENV" "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a"
fi

cat <<EOF

Environment ready: $VENV

Start the demo:
  ./run.sh

The model weights are downloaded on first launch unless DA3_MODEL_DIR points to a local checkpoint.
EOF
