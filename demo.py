#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import ExifTags, Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
VERSION = "0.2.0"
DEPTH_ANYTHING_SRC = ROOT / "Depth-Anything-3" / "src"

if str(DEPTH_ANYTHING_SRC) not in sys.path:
    sys.path.insert(0, str(DEPTH_ANYTHING_SRC))

from depth_anything_3.api import DepthAnything3  # noqa: E402

SKIP_LABELS = {"floor", "ceiling", "wall", "ground", "sky", "room", "space", "area"}
MARKER_COLORS = {
    "chair": "#ff4444",
    "table": "#44ff44",
    "door": "#4444ff",
    "person": "#ff8800",
    "plant": "#00cc44",
    "monitor": "#00ccff",
    "lamp": "#ffff00",
    "window": "#88ccff",
    "couch": "#cc44cc",
    "bed": "#ff6688",
    "sink": "#44cccc",
    "toilet": "#cccc44",
    "tv": "#0088ff",
    "book": "#cc8844",
    "bottle": "#44ccaa",
    "cup": "#ffaa44",
    "keyboard": "#aaaaaa",
    "phone": "#88ff88",
    "shelf": "#886644",
    "box": "#ff44aa",
    "cabinet": "#668844",
}
TARGET_PALETTE = ["#ff4444", "#44ff44", "#4444ff", "#ff8800", "#00ccff", "#cc44cc", "#ffff00", "#44cccc"]
DEFAULT_FOV_DEG = 60.0
DEPTH_MODELS = {
    "da3": "Depth Anything 3 (DA3METRIC-LARGE)",
    "moge3-l": "MoGe-3 ViT-L",
    "moge3-g": "MoGe-3 ViT-G",
}
MOGE_REPOS = {"moge3-l": "Ruicheng/moge-3-vitl", "moge3-g": "Ruicheng/moge-3-vitg"}
# Molmo2 can't follow the JSON prompt, so a multi-target prompt is split into one "Point to ..." query per phrase.
PROMPT_VERB = re.compile(r"^(?:please\s+)?(?:select|find|locate|mark|show(?:\s+me)?|point\s+to)\s+", re.IGNORECASE)
TARGET_SEPARATOR = re.compile(r"\s*,\s*(?:and\s+)?|\s+and\s+", re.IGNORECASE)
LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
MOLMO_COORDS = re.compile(r'coords="([^"]+)')
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
# Per VLM call (Molmo2 makes one per target). A cold first request in eager mode adds 1-3 s (measured on the RTX PRO 6000 server and Orin).
VLM_TIMEOUT_S = float(os.environ.get("VLM_TIMEOUT", "45"))
# Multipart boundaries, headers and the prompt field on top of the image.
MAX_INFER_BODY_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024


def exif_fov_deg(image: Image.Image) -> float | None:
    """Field of view across the long side from the EXIF 35 mm-equivalent focal length, or None."""
    exif = image.getexif().get_ifd(ExifTags.IFD.Exif)
    try:
        focal_35mm = float(exif.get(ExifTags.Base.FocalLengthIn35mmFilm, 0))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(focal_35mm) or focal_35mm <= 0:
        return None
    width, height = image.size
    # A resize keeps the equivalence, a crop doesn't: skip when the pixels' aspect differs from the original's.
    exif_size = (exif.get(ExifTags.Base.ExifImageWidth), exif.get(ExifTags.Base.ExifImageHeight))
    if any(v is not None for v in exif_size):
        try:
            exif_w, exif_h = (float(v) for v in exif_size)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None  # malformed EXIF: can't tell a crop from a resize, so don't trust the focal length
        if not all(math.isfinite(v) and v > 0 for v in (exif_w, exif_h)):
            return None
        if abs(
            max(exif_w, exif_h) / min(exif_w, exif_h) - max(width, height) / min(width, height)
        ) > 0.02:
            return None
    # The 35 mm equivalence is defined on the diagonal (36x24 mm), so it holds for any aspect ratio or orientation.
    focal_px = focal_35mm * math.hypot(width, height) / math.hypot(36.0, 24.0)
    return math.degrees(2.0 * math.atan(max(width, height) / 2.0 / focal_px))


def estimate_focal_px(width: int, height: int, fov_deg: float) -> float:
    fov_rad = math.radians(fov_deg)
    if not 0 < fov_rad < math.pi:
        raise ValueError("fov_deg must be between 0 and 180.")
    sensor_span = float(max(width, height))
    return 0.5 * sensor_span / math.tan(fov_rad / 2.0)


def depth_preview_base64(depth_m: np.ndarray, valid_mask: np.ndarray) -> str:
    if not np.any(valid_mask):
        raise ValueError("No valid depth values available for preview.")

    lo, hi = np.percentile(depth_m[valid_mask], [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1e-6

    scaled = np.clip((np.where(valid_mask, depth_m, lo) - lo) / (hi - lo), 0.0, 1.0)
    preview = (scaled * 255.0).astype(np.uint8)
    return image_base64(Image.fromarray(preview))


def float16_base64(values: np.ndarray) -> str:
    # Half precision: ~0.05% relative (4 mm at 5 m, 25 cm at 500 m), half the bytes of float32.
    return base64.b64encode(np.ascontiguousarray(values, dtype="<f2").tobytes()).decode("ascii")


def image_base64(image: Image.Image, format: str = "PNG") -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format=format, quality=92)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def base_marker_label(label: str) -> str:
    return re.sub(r"[_ ]\d+$", "", label.strip().lower())


def marker_color_hex(label: str, index: int) -> str:
    return MARKER_COLORS.get(base_marker_label(label), TARGET_PALETTE[index % len(TARGET_PALETTE)])


def normalize_vlm_url(url: str) -> str:
    cleaned = url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/v1/chat/completions"


def parse_vlm_models(spec: str | None, default_url: str, default_model: str) -> dict[str, str]:
    """VLM_MODELS "name=url,name=url" (name = the served model name) -> {name: chat URL}; else the one default VLM."""
    if not spec or not spec.strip():
        return {default_model: normalize_vlm_url(default_url)}
    models = {}
    for item in spec.split(","):
        name, sep, url = item.strip().partition("=")
        if not sep or not name.strip() or not url.strip():
            raise ValueError(f"VLM_MODELS entries must be name=url, got {item!r}")
        models[name.strip()] = normalize_vlm_url(url.strip())
    return models


def is_molmo(vlm: str) -> bool:
    return "molmo" in vlm.lower()


def moge_available(device: str) -> bool:
    if not device.startswith("cuda"):
        return False  # MoGe-3 needs Triton (FlexGEMM): Linux + CUDA only.
    try:
        import moge.model.v3  # noqa: F401
    except ImportError:
        return False
    return True


def split_target_phrases(prompt: str) -> list[str]:
    text = PROMPT_VERB.sub("", prompt.strip()).rstrip(" .!?")
    phrases = [LEADING_ARTICLE.sub("", part.strip()) for part in TARGET_SEPARATOR.split(text)]
    return [p for p in phrases if p][:8]


def parse_molmo_points(raw: str) -> list[tuple[float, float]]:
    """Points from Molmo2's <points coords="..."> (0-1000 grid); a missing closing quote (max_tokens) is fine."""
    match = MOLMO_COORDS.search(raw)
    if not match:
        return []
    values = match.group(1).split()
    # Triples "index x y", after a 1-based image index from vLLM ("1 1 770 483"); coordinates are zero-padded
    # ("018"), so a leading "1 1" can only be image 1, point 1.
    start = 1 if values[:2] == ["1", "1"] else 0
    points = []
    for i in range(start, len(values) - 2, 3):
        try:
            x, y = float(values[i + 1]), float(values[i + 2])
        except ValueError:
            break
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    return points


def confidence_text(confidence: float | None) -> str:
    return "" if confidence is None else f" {confidence * 100:.0f}%"


def valid_depth_mask(depth_m: np.ndarray, sky_mask: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    """Finite, positive, non-sky depth inside the 1st-99th percentile; returns (mask, near, far)."""
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if sky_mask is not None:
        valid &= ~sky_mask.astype(bool)

    if not np.any(valid):
        raise ValueError("No valid depth pixels remained after filtering.")

    near, far = np.percentile(depth_m[valid], [1.0, 99.0])
    valid &= depth_m >= near
    valid &= depth_m <= far
    return valid, float(near), float(far)


def custom_prompt_template(user_prompt: str) -> str:
    return f"""You are a vision assistant that localizes user-requested targets in a single image.
Find the visible image locations that best satisfy this request:
{user_prompt}

Return ONLY valid JSON in this exact format:
{{"targets":[{{"label":"short lowercase label","x":320,"y":650,"confidence":0.9}}]}}

Coordinate rules:
- x and y must be integers in the range 0-1000
- (0,0) is the top-left of the image
- (1000,1000) is the bottom-right of the image

Content rules:
- Include only targets that are clearly relevant to the request
- Use short lowercase labels
- Maximum 8 targets
- If nothing relevant is visible, return {{"targets":[]}}
- No markdown, no prose, no code fences, only JSON"""


def parse_vlm_targets(raw: str) -> list[dict]:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    else:
        brace = cleaned.find("{")
        if brace >= 0:
            cleaned = cleaned[brace:]

    # Model output is untrusted: malformed JSON or odd values mean "no targets", not a failed request.
    try:
        data, _ = json.JSONDecoder().raw_decode(cleaned)
    except json.JSONDecodeError:
        return []
    targets = data.get("targets", []) if isinstance(data, dict) else []
    valid = []
    for obj in targets if isinstance(targets, list) else []:
        if not isinstance(obj, dict) or not all(k in obj for k in ("label", "x", "y")):
            continue
        label = str(obj["label"]).lower().strip()
        if not label or base_marker_label(label) in SKIP_LABELS:
            continue
        try:
            x, y = int(obj["x"]), int(obj["y"])
            confidence = float(obj.get("confidence", 0.7))
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(confidence):
            continue
        valid.append(
            {
                "label": label,
                "x": min(1000, max(0, x)),
                "y": min(1000, max(0, y)),
                "confidence": min(1.0, max(0.0, confidence)),
            }
        )
        if len(valid) == 8:
            break
    for index, target in enumerate(valid):
        target["color"] = marker_color_hex(target["label"], index)
    return valid


def annotate_targets(rgb: np.ndarray, targets: list[dict]) -> str:
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size

    for target in targets:
        px = int(np.clip(target["x"] / 1000.0 * w, 0, w - 1))
        py = int(np.clip(target["y"] / 1000.0 * h, 0, h - 1))
        color = target["color"]
        r = max(6, min(w, h) // 60)
        draw.ellipse((px - r, py - r, px + r, py + r), outline=color, width=3)
        draw.line((px, py, px, py - (r * 3)), fill=color, width=3)
        text = f'{target["label"]}{confidence_text(target["confidence"])}'
        tx = min(max(8, px + r + 6), max(8, w - 140))
        ty = max(8, py - (r * 3) - 18)
        draw.rounded_rectangle((tx - 6, ty - 4, tx + 130, ty + 18), radius=6, fill=(0, 0, 0, 190), outline=color)
        draw.text((tx, ty), text, fill="white")

    return image_base64(image)


def project_target_to_3d(
    target: dict,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> dict | None:
    img_h, img_w = depth_map.shape
    px = int(np.clip(target["x"] / 1000.0 * img_w, 0, img_w - 1))
    py = int(np.clip(target["y"] / 1000.0 * img_h, 0, img_h - 1))

    r = 5
    y0, y1 = max(0, py - r), min(img_h, py + r + 1)
    x0, x1 = max(0, px - r), min(img_w, px + r + 1)
    patch = depth_map[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.15)]
    if valid.size == 0:
        return None

    depth_m = float(np.median(valid))
    x_cam = (px - cx) * depth_m / fx
    y_cam = -(py - cy) * depth_m / fy
    z_cam = -depth_m
    return {
        "label": target["label"],
        "color": target["color"],
        "confidence": target["confidence"],
        "pixel": {"x": int(px), "y": int(py)},
        "position": {
            "x": float(x_cam),
            "y": float(y_cam),
            "z": float(z_cam),
        },
    }


class DemoRuntime:
    def __init__(self, model_dir: str, device: str, process_res: int, default_depth_model: str = "da3"):
        self.model_dir = os.path.expanduser(model_dir)
        self.device = device
        self.process_res = process_res
        self.vlms = parse_vlm_models(
            os.environ.get("VLM_MODELS"),
            os.environ.get("QWEN_URL", "http://127.0.0.1:8012/v1"),
            os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
        )
        self.depth_models = ["da3"] + ([m for m in MOGE_REPOS] if moge_available(device) else [])
        if default_depth_model not in self.depth_models:
            raise ValueError(f"Depth model {default_depth_model!r} unavailable here; available: {self.depth_models}")
        self.default_depth_model = default_depth_model
        self._model: DepthAnything3 | None = None
        self._moge: dict = {}
        self._lock = threading.Lock()

    def load_model(self) -> DepthAnything3:
        with self._lock:
            if self._model is None:
                print(f"Loading model: {self.model_dir} on {self.device}")
                model_path = Path(self.model_dir)
                if self.model_dir.startswith(("/", ".", "~")) and not model_path.is_dir():
                    raise FileNotFoundError(
                        "Local DA3 model path must be a directory containing Hugging Face-style "
                        f"weights; got: {model_path}"
                    )
                self._model = DepthAnything3.from_pretrained(self.model_dir).to(self.device).eval()
            return self._model

    def load_moge(self, name: str):
        path = None
        if name not in self._moge:
            from huggingface_hub import hf_hub_download

            # Outside the lock: a first download can take minutes and would stall every other request's forward pass.
            path = hf_hub_download(MOGE_REPOS[name], "model.pt")
        with self._lock:
            if name not in self._moge:
                from moge.model.v3 import MoGeModel

                print(f"Loading model: {MOGE_REPOS[name]} on {self.device}")
                self._moge[name] = MoGeModel.from_pretrained(path).to(self.device).eval()
            return self._moge[name]

    def predict_depth(self, depth_model: str, image: Image.Image, fov_deg: float | None):
        """-> (rgb, metric depth, invalid mask, focal px, long-side FOV, FOV source) at the processing resolution."""
        if depth_model == "da3":
            model = self.load_model()
            # One GPU forward at a time keeps memory bounded; the slow VLM call stays concurrent.
            with self._lock:
                prediction = model.inference(
                    [np.asarray(image)],
                    process_res=self.process_res,
                    process_res_method="upper_bound_resize",
                )
            rgb = prediction.processed_images[0]
            raw_depth = prediction.depth[0].astype(np.float32)
            invalid = prediction.sky[0].astype(bool) if prediction.sky is not None else np.zeros(raw_depth.shape, bool)
            height, width = raw_depth.shape
            used_fov_deg = fov_deg or DEFAULT_FOV_DEG
            focal_px = estimate_focal_px(width, height, used_fov_deg)
            return rgb, raw_depth * (focal_px / 300.0), invalid, focal_px, used_fov_deg, "exif" if fov_deg else "assumed"

        model = self.load_moge(depth_model)
        scale = self.process_res / max(image.size)
        if scale < 1:
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
        rgb = np.asarray(image)
        height, width = rgb.shape[:2]
        fov_x = None
        if fov_deg:
            # The EXIF FOV spans the long side; MoGe takes the horizontal one.
            fov_x = math.degrees(2 * math.atan(width / 2 / estimate_focal_px(width, height, fov_deg)))
        tensor = torch.tensor(rgb, dtype=torch.float32, device=self.device).permute(2, 0, 1).div_(255.0)
        with self._lock, torch.no_grad():
            # fp16: 1.8x (ViT-L) / 2.7x (ViT-G) faster on the RTX PRO 6000 server, 2.2x on Orin; depth within 0.04% median, same masks/FOV.
            output = model.infer(tensor, fov_x=fov_x, use_fp16=True)
        depth = output["depth"].float().cpu().numpy()
        valid = output["mask"].bool().cpu().numpy() & np.isfinite(depth)
        # MoGe intrinsics are normalized by image width/height; square pixels, so one focal length.
        focal_px = float(output["intrinsics"][0, 0]) * width
        long_fov_deg = math.degrees(2 * math.atan(max(width, height) / 2 / focal_px))
        metric_depth_m = np.where(valid, depth, 0.0).astype(np.float32)
        return rgb, metric_depth_m, ~valid, focal_px, long_fov_deg, "exif" if fov_deg else "estimated"

    def infer_image(
        self,
        image: Image.Image,
        prompt: str | None = None,
        fov_deg: float | None = None,
        depth_model: str | None = None,
        vlm: str | None = None,
    ) -> dict:
        depth_model = depth_model or self.default_depth_model
        vlm = vlm or next(iter(self.vlms))
        prompt = prompt.strip() if prompt else None
        rgb, metric_depth_m, invalid, focal_px, used_fov_deg, fov_source = self.predict_depth(
            depth_model, image.convert("RGB"), fov_deg
        )

        height, width = metric_depth_m.shape
        # Apply model validity to previews and target patches too; float16 cannot represent depths beyond 65 km.
        preview_mask = ~invalid & np.isfinite(metric_depth_m) & (metric_depth_m > 0) & (metric_depth_m <= 65504)
        metric_depth_m = np.where(preview_mask, metric_depth_m, 0.0)
        valid, near, far = valid_depth_mask(metric_depth_m)

        targets_2d = []
        targets_3d = []
        overlay_base64 = None
        if prompt:
            targets_2d = self.query_vlm_targets(Image.fromarray(rgb, mode="RGB"), prompt, vlm)
            overlay_base64 = annotate_targets(rgb, targets_2d)
            for target in targets_2d:
                projected = project_target_to_3d(
                    target=target,
                    depth_map=metric_depth_m,
                    fx=focal_px,
                    fy=focal_px,
                    cx=width * 0.5,
                    cy=height * 0.5,
                )
                if projected is not None:
                    targets_3d.append(projected)

        return {
            # The browser builds the point cloud and the splats from these: full-resolution depth (0 = invalid)
            # and the image it was predicted on.
            "depth": float16_base64(np.where(valid, metric_depth_m, 0.0)),
            "rgb": image_base64(Image.fromarray(rgb, mode="RGB"), "JPEG"),
            "depth_preview": depth_preview_base64(metric_depth_m, preview_mask),
            "annotated_preview": overlay_base64,
            "targets_2d": targets_2d,
            "targets_3d": targets_3d,
            "meta": {
                "depth_model": depth_model,
                "depth_model_label": DEPTH_MODELS[depth_model],
                "device": self.device,
                "vlm_model": vlm if prompt else None,
                "input_size": {"width": int(image.width), "height": int(image.height)},
                "processed_size": {"width": int(width), "height": int(height)},
                "focal_px": float(focal_px),
                "fov_deg": float(used_fov_deg),
                "fov_source": fov_source,
                "is_metric_model_output": True,
                "prompt_used": prompt,
                "target_count": len(targets_3d),
                "depth_near_m": near,
                "depth_far_m": far,
            },
        }

    def query_vlm_targets(self, image: Image.Image, prompt: str, vlm: str) -> list[dict]:
        w, h = image.size
        max_dim = 384
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

        image_b64 = image_base64(image)
        if not is_molmo(vlm):
            return parse_vlm_targets(self.ask_vlm(vlm, custom_prompt_template(prompt), image_b64))

        targets = []
        for phrase in split_target_phrases(prompt):
            if len(targets) == 8:
                break
            if base_marker_label(phrase.lower()) in SKIP_LABELS:
                continue
            points = parse_molmo_points(self.ask_vlm(vlm, f"Point to the {phrase}.", image_b64))
            for x, y in points[:8 - len(targets)]:
                # Molmo2 points carry no confidence.
                targets.append({"label": phrase.lower(), "x": int(np.clip(x, 0, 1000)), "y": int(np.clip(y, 0, 1000)),
                                "confidence": None})
        for index, target in enumerate(targets):
            target["color"] = marker_color_hex(target["label"], index)
        return targets

    def ask_vlm(self, vlm: str, text: str, image_b64: str) -> str:
        payload = {
            "model": vlm,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 400,
            "temperature": 0,
            # Qwen3.5/3.6 think by default, which only adds latency here; models without the switch ignore it.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = requests.post(self.vlms[vlm], json=payload, timeout=VLM_TIMEOUT_S)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] or ""


def build_app(runtime: DemoRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if runtime.default_depth_model == "da3":
            runtime.load_model()
        else:
            runtime.load_moge(runtime.default_depth_model)
        yield

    app = FastAPI(title="Apollo Spatial VLM", version=VERSION, lifespan=lifespan)

    @app.middleware("http")
    async def limit_infer_body(request: Request, call_next):
        # Reject before multipart parsing spools the body; uvicorn never reads past Content-Length.
        if request.url.path == "/api/infer":
            length = request.headers.get("content-length")
            if length is None:
                return JSONResponse({"detail": "Content-Length required."}, status_code=411)
            if not (length.isascii() and length.isdigit()) or int(length) > MAX_INFER_BODY_BYTES:
                return JSONResponse(
                    {"detail": f"Image larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."}, status_code=413
                )
        return await call_next(request)

    @app.get("/")
    async def index() -> FileResponse:
        response = FileResponse(ROOT / "index.html")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    @app.get("/index.js")
    async def index_js() -> FileResponse:
        response = FileResponse(ROOT / "index.js", media_type="application/javascript")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "ok": True,
            "version": VERSION,
            "device": runtime.device,
            "model_dir": runtime.model_dir,
            "depth_models": runtime.depth_models,
            "depth_models_loaded": (["da3"] if runtime._model is not None else []) + list(runtime._moge),
            "vlms": runtime.vlms,
            "model_loaded": runtime._model is not None or bool(runtime._moge),
        }

    @app.get("/api/models")
    async def models() -> dict:
        return {
            "depth": [{"id": m, "label": DEPTH_MODELS[m]} for m in runtime.depth_models],
            "default_depth": runtime.default_depth_model,
            "vlm": list(runtime.vlms),
            "default_vlm": next(iter(runtime.vlms)),
        }

    @app.post("/api/infer")
    def infer(
        image: UploadFile = File(...),
        prompt: str | None = Form(default=None),
        depth_model: str | None = Form(default=None),
        vlm: str | None = Form(default=None),
    ) -> dict:
        if depth_model and depth_model not in runtime.depth_models:
            raise HTTPException(status_code=400, detail=f"Unknown depth model {depth_model!r}; available: {runtime.depth_models}")
        if vlm and vlm not in runtime.vlms:
            raise HTTPException(status_code=400, detail=f"Unknown VLM {vlm!r}; available: {list(runtime.vlms)}")
        if image.size is not None and image.size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Image larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        try:
            payload = image.file.read()
            pil_image = Image.open(io.BytesIO(payload))
            fov_deg = exif_fov_deg(pil_image)
            pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded image: {exc}") from exc

        try:
            return runtime.infer_image(
                image=pil_image,
                prompt=prompt,
                fov_deg=fov_deg,
                depth_model=depth_model,
                vlm=vlm,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth Anything 3 metric point-cloud demo server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model-dir", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--depth-model", choices=list(DEPTH_MODELS), default="da3", help="default for the dropdown")
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = DemoRuntime(
        model_dir=args.model_dir,
        device=args.device,
        process_res=args.process_res,
        default_depth_model=args.depth_model,
    )
    app = build_app(runtime)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
