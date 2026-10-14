#!/usr/bin/env bash
# Serve one VLM for eval/vlm_eval.py on one GPU, loopback only.
# Usage: eval/serve.sh GPU PORT NAME MODEL [extra vLLM args...]
#   eval/serve.sh 2 8013 molmo2-8b allenai/Molmo2-8B
# Weights must already be downloaded: HF_HUB_OFFLINE avoids Hugging Face calls.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="$1" PORT="$2" NAME="$3" MODEL="$4"
shift 4

CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 exec "$REPO_DIR/.venv-vlm/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9 \
  "$@"
