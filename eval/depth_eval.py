"""Metric depth accuracy on DIODE val: DA3METRIC-LARGE (the demo's pipeline) vs MoGe-3.

DIODE's camera has a 60.0° horizontal FOV, the same as the demo's guess, so full frames flatter DA3. Centre crops
narrow the FOV with known ground truth, which is where a FOV guess goes wrong and an estimated FOV should help.

Usage (repo root):
  .venv/bin/python eval/depth_eval.py --model da3            # demo pipeline: 60° FOV guess
  .venv/bin/python eval/depth_eval.py --model da3-gt-focal   # DA3 with the true focal length (what EXIF would give)
  eval/.venv-moge/bin/python eval/depth_eval.py --model moge3-l   # or moge3-g
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "eval" / "data"
# DIODE devkit intrinsics.txt: [fx, fy, cx, cy] for the 1024x768 frames.
DIODE_FX, DIODE_FY = 886.81, 927.06
# Same patch as the demo's harbour-bridge target (normalized 0-1000 VLM point) on the SOH sample.
SOH_IMAGE = REPO / "Depth-Anything-3" / "assets" / "examples" / "SOH" / "000.png"
SOH_BRIDGE = (780, 420)


def hfov_deg(fx: float, width: float) -> float:
    return math.degrees(2 * math.atan(width / 2 / fx))


class DA3:
    def __init__(self, gt_focal: bool):
        sys.path.insert(0, str(REPO))
        from demo import DEFAULT_FOV_DEG, DepthAnything3, estimate_focal_px

        self.gt_focal = gt_focal
        self.fov_guess, self.estimate_focal_px = DEFAULT_FOV_DEG, estimate_focal_px
        self.model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to("cuda").eval()

    def __call__(self, rgb: np.ndarray, focal_px: float | None) -> tuple[np.ndarray, float | None]:
        prediction = self.model.inference([rgb], process_res=504, process_res_method="upper_bound_resize")
        raw = prediction.depth[0].astype(np.float32)
        h, w = raw.shape
        if self.gt_focal and focal_px is not None:
            focal = focal_px * w / rgb.shape[1]  # true focal, rescaled to the processed resolution
        else:
            focal = self.estimate_focal_px(w, h, self.fov_guess)  # demo.py
        return raw * (focal / 300.0), None


class MoGe3:
    def __init__(self, size: str):
        from huggingface_hub import hf_hub_download
        from moge.model.v3 import MoGeModel

        path = hf_hub_download(f"Ruicheng/moge-3-vit{size}", "model.pt")
        self.model = MoGeModel.from_pretrained(path).to("cuda").eval()

    def __call__(self, rgb: np.ndarray, focal_px: float | None) -> tuple[np.ndarray, float | None]:
        image = torch.tensor(rgb / 255.0, dtype=torch.float32, device="cuda").permute(2, 0, 1)
        output = self.model.infer(image)
        depth = output["depth"].float().cpu().numpy()
        fx = float(output["intrinsics"][0, 0])
        # MoGe intrinsics are normalized by image size.
        fov = hfov_deg(fx * rgb.shape[1], rgb.shape[1]) if fx < 10 else hfov_deg(fx, rgb.shape[1])
        return depth, fov


def centre_crop(rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray, frac: float):
    if frac >= 1.0:
        return rgb, depth, mask
    h, w = mask.shape
    ch, cw = int(h * frac), int(w * frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return rgb[y0:y0 + ch, x0:x0 + cw], depth[y0:y0 + ch, x0:x0 + cw], mask[y0:y0 + ch, x0:x0 + cw]


def metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    p, g = pred[valid], gt[valid]
    ratio = np.maximum(p / g, g / p)
    aligned = p * (np.median(g) / np.median(p))  # scale-only alignment: shape quality without the metric scale
    ratio_aligned = np.maximum(aligned / g, g / aligned)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "delta1": float(np.mean(ratio < 1.25)),
        "scale_ratio": float(np.median(p) / np.median(g)),
        "abs_rel_aligned": float(np.mean(np.abs(aligned - g) / g)),
        "delta1_aligned": float(np.mean(ratio_aligned < 1.25)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["da3", "da3-gt-focal", "moge3-l", "moge3-g"])
    parser.add_argument("--crops", default="1.0,0.5", help="centre-crop fractions; 1.0 = full frame")
    parser.add_argument("--limit", type=int, default=0, help="first N frames only (smoke test)")
    args = parser.parse_args()

    if args.model.startswith("da3"):
        model = DA3(gt_focal=args.model == "da3-gt-focal")
    else:
        model = MoGe3(args.model[-1])

    frames = sorted((DATA / "val").glob("*/*/*/*_depth.npy"))
    if args.limit:
        frames = frames[:: max(1, len(frames) // args.limit)][: args.limit]
    crops = [float(c) for c in args.crops.split(",")]

    rows, latencies = [], []
    for i, depth_path in enumerate(frames):
        scene = "outdoor" if "/outdoor/" in str(depth_path) else "indoor"
        rgb_full = np.asarray(Image.open(str(depth_path).replace("_depth.npy", ".png")).convert("RGB"))
        gt_full = np.load(depth_path)[..., 0]
        mask_full = np.load(str(depth_path).replace("_depth.npy", "_depth_mask.npy")) > 0
        for frac in crops:
            rgb, gt, mask = centre_crop(rgb_full, gt_full, mask_full, frac)
            torch.cuda.synchronize()
            started = time.perf_counter()
            pred, fov = model(rgb, focal_px=(DIODE_FX + DIODE_FY) / 2)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
            if pred.shape != gt.shape:
                pred = np.asarray(Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.BILINEAR))
            valid = mask & (gt > 0) & np.isfinite(pred) & (pred > 0)
            if valid.sum() < 100:
                continue
            row = {"frame": depth_path.name, "scene": scene, "crop": frac, **metrics(pred, gt, valid)}
            row["gt_hfov"] = hfov_deg(DIODE_FX, rgb.shape[1])
            if fov is not None:
                row["pred_hfov"] = fov
            rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(frames)}]", flush=True)

    summary = {"model": args.model, "frames": len(frames), "latency_median_s": round(statistics.median(latencies), 4)}
    for frac in crops:
        for scene in ("indoor", "outdoor", "all"):
            group = [r for r in rows if r["crop"] == frac and scene in ("all", r["scene"])]
            if not group:
                continue
            key = f"crop{frac:g}/{scene}"
            summary[key] = {m: round(statistics.mean(r[m] for r in group), 4)
                            for m in ("abs_rel", "delta1", "scale_ratio", "abs_rel_aligned", "delta1_aligned")}
            summary[key]["n"] = len(group)
            summary[key]["gt_hfov"] = round(group[0]["gt_hfov"], 1)
            if "pred_hfov" in group[0]:
                summary[key]["pred_hfov_mean"] = round(statistics.mean(r["pred_hfov"] for r in group), 1)
                summary[key]["hfov_abs_err_mean"] = round(
                    statistics.mean(abs(r["pred_hfov"] - r["gt_hfov"]) for r in group), 1)

    soh = np.asarray(Image.open(SOH_IMAGE).convert("RGB"))
    soh_depth, _ = model(soh, focal_px=None)
    h, w = soh_depth.shape
    px, py = int(SOH_BRIDGE[0] / 1000 * w), int(SOH_BRIDGE[1] / 1000 * h)
    patch = soh_depth[max(0, py - 5):py + 6, max(0, px - 5):px + 6]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    # None when the model masks the bridge as invalid (MoGe marks far/sky pixels inf).
    summary["soh_bridge_depth_m"] = round(float(np.median(patch)), 2) if patch.size else None

    out = REPO / "eval" / "results" / f"depth-{args.model}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
