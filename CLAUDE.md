# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Apollo Spatial VLM, a private fork of [MercuriusTech/Odyseus-Spatial-VLM](https://github.com/MercuriusTech/Odyseus-Spatial-VLM) (renamed 2026-10-14; `upstream` remote still points there). A single-page demo: user uploads an image and a prompt, a VLM picks 2D targets, Depth Anything 3 (DA3) produces metric depth, and targets are lifted to 3D and shown in a Three.js point-cloud viewer. All app code is three files: `demo.py` (FastAPI backend), `index.html` + `index.js` (frontend, no build step; Three.js 0.186.1 and Spark 2.3.1 loaded from unpkg via import map; Spark needs Three.js >= 0.180).

`Depth-Anything-3/` is a git submodule of upstream ByteDance-Seed/Depth-Anything-3. Don't edit it; `demo.py` puts `Depth-Anything-3/src` on `sys.path` and imports `depth_anything_3.api.DepthAnything3`. After cloning: `git submodule update --init --recursive`.

## Two processes

1. **VLM server**: any OpenAI-compatible `/v1/chat/completions` endpoint serving a vision model. Reference is vLLM with `Qwen/Qwen3-VL-8B-Instruct` on port 8012. `demo.py` reads `QWEN_URL` (default `http://127.0.0.1:8012/v1`; `normalize_vlm_url` appends the path) and `VLM_MODEL`.
2. **Demo server**: `./run.sh` → `.venv/bin/python demo.py --host --port 8080 --model-dir depth-anything/DA3METRIC-LARGE --process-res 504 [--device]` (device defaults to cuda, then mps, then cpu). The DA3 model loads in the FastAPI lifespan, so startup blocks until weights are downloaded/loaded. `--model-dir` can be an HF repo id or a local HF-style directory (paths starting `/`, `.`, `~`).

Endpoints: `GET /` and `/index.js` (served no-cache), `GET /healthz` (device, VLM URL/model, model_loaded), `POST /api/infer` (multipart `image`, optional `prompt`).

## Pipeline in `DemoRuntime.infer_image`

- DA3 inference at `process_res` (upper-bound resize). Depth is rescaled to "metric" with an assumed camera: fixed `DEFAULT_FOV_DEG = 60`, `focal = estimate_focal_px(...)`, `metric = raw_depth * focal / 300`. No real intrinsics are used, so absolute distances are only plausible for ~60° FOV photos.
- Point cloud: sky mask and 1st/99th depth percentile removed, uniformly subsampled to `DEFAULT_MAX_POINTS = 50000`. Camera frame is Three.js-style: x right, y up, z = -depth.
- VLM: image downscaled to max 384 px, prompt from `custom_prompt_template` demands strict JSON `{"targets":[{label,x,y,confidence}]}` with **x/y normalized 0–1000** (Qwen3-VL's native grounding convention). `parse_vlm_targets` strips `<think>` blocks and code fences, drops `SKIP_LABELS` (floor, wall, sky…), caps at 8.
- Colors: `parse_vlm_targets` assigns each target a `color` (fixed per known label, else palette by index). The 2D overlay, 3D markers and target list all use that field; the frontend has no color logic.
- 3D lift: `project_target_to_3d` takes the median depth of an 11×11 patch (> 0.15 m); a target with no valid depth there is dropped from `targets_3d`.
- Response: `points`/`colors` are base64 little-endian float32 (N×3), plus base64 PNGs (depth preview, annotated overlay). `depth` is the full-resolution metric depth grid (float32, `processed_size` H×W, 0 = invalid, same mask as the points) and `rgb` the processed image as JPEG. `index.js` decodes these by field name; change both sides together.
- Splats (`loadSplats` in `index.js`): the browser builds one flat Gaussian per valid depth pixel with Spark, using the same camera model as `demo.py`. Splats are oriented from neighbouring depths and fall back to camera-facing discs across depth edges (`SPLAT_EDGE`) and past `SPLAT_MAX_STRETCH` footprints. The "Gaussian splats" checkbox toggles between splats and points.
- `/api/infer` is a sync handler on purpose (FastAPI threadpool), so a slow VLM call doesn't block other requests; the DA3 forward pass is serialized by `DemoRuntime._lock`. Uploads over `MAX_UPLOAD_BYTES` (20 MB) get 413.
- Frontend renders VLM labels only via `textContent` (labels are model output, steerable by prompt/image text). Images without `src` are hidden by CSS and a sibling `.placeholder` shows instead; reset an image with `removeAttribute("src")`, never `src = ""`.

## Setup and run

All environments are uv venvs, no conda: `./setup.sh` → `.venv` (demo; `TORCH_BACKEND=auto` picks the torch build for the driver), `./run.sh`; `./setup-vlm.sh` → `.venv-vlm` (vLLM, pinned `VLLM_VERSION`, Linux only, separate because vLLM pins its own torch) and writes `.env.vlm`; `./run-vlm.sh` sources `.env.vlm` and starts vLLM in tmux. `setup-vlm.ps1` only writes `.env.vlm`.

Platform facts (verified):
- DA3 imports `pycolmap` at load time (`utils/export/colmap.py`) though the demo never exports COLMAP. No Linux aarch64 wheel: build COLMAP from source and `uv pip install` the COLMAP repo root (pycolmap lives there); `setup.sh` stops with that hint.
- `xformers` is optional (DA3 falls back in `swiglu_ffn.py`); don't add it unpinned, it drags torch to a newer build.
- macOS: vLLM doesn't run; use Ollama (`QWEN_URL=http://127.0.0.1:11434/v1`, `VLM_MODEL=qwen3-vl:8b-instruct`). `run.sh` sets `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Jetson AGX Orin, JetPack 7 / CUDA 13 (sm_87): jetson-ai-lab `sbsa/cu130` wheels only ship sm_110/sm_121 ("no kernel image"). PyPI vLLM (0.31) has sm_87 kernels; PyPI torch runs DA3 correctly via sm_80 binaries (depth matched MPS to ~0.3% median). Unified memory: use `GPU_UTIL=0.5`.

## Verifying changes

Unit tests (parser, colors, float32 encoding; imports `demo`, so run from the repo root with the venv): `.venv/bin/python -m unittest discover -s tests -v`. Single test: add `-k test_name`. No linter config. End to end: `curl localhost:8080/healthz`, then
`curl -X POST localhost:8080/api/infer -F image=@Depth-Anything-3/assets/examples/SOH/000.png -F "prompt=select the opera house, the harbour bridge and the red buoy"`
and check `targets_2d`/`targets_3d` and the decoded `annotated_preview`.
