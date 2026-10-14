"""Point-in-box accuracy of a VLM on the RefCOCO sample (eval/prepare_refcoco.py).

Qwen-style models go through the demo's own prompt, image encoding and parser, so the score is what the demo
would show. Molmo2 is asked natively ("Point to ...") and answers with <points coords="1 x y">, also 0-1000.

Usage (repo root, demo venv):
  .venv/bin/python eval/vlm_eval.py --url http://127.0.0.1:8012/v1 --model <served name> --name qwen3-vl-32b
  add --kind molmo for Molmo2, --no-think for Qwen3.5/3.6 (thinking is on by default there),
  --prompt native to ask Qwen for its own grounding format (a bbox_2d box; the centre is scored)
"""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import requests
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo import custom_prompt_template, image_base64, normalize_vlm_url, parse_vlm_targets  # noqa: E402

DATA = REPO / "eval" / "data"
# No closing quote required: a long list of points can hit max_tokens mid-attribute, and only the first is used.
MOLMO_COORDS = re.compile(r'coords="([^"]+)')
QWEN_BOX = re.compile(r'"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]')


def resize(image: Image.Image, max_dim: int) -> Image.Image:
    # Same downscale as DemoRuntime.query_vlm_targets.
    w, h = image.size
    if max(w, h) <= max_dim:
        return image
    scale = max_dim / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def ask(url: str, model: str, text: str, image: Image.Image, no_think: bool) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64(image)}"}},
        ]}],
        "max_tokens": 400,
        "temperature": 0,
    }
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    response = requests.post(url, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"] or ""


def point_qwen(raw: str) -> tuple[float, float] | None:
    targets = parse_vlm_targets(raw)
    if not targets:
        return None
    best = max(targets, key=lambda t: t["confidence"])
    return best["x"], best["y"]


def point_qwen_native(raw: str) -> tuple[float, float] | None:
    match = QWEN_BOX.search(raw)
    if not match:
        return None
    x0, y0, x1, y1 = (float(v) for v in match.groups())
    return (x0 + x1) / 2, (y0 + y1) / 2


def point_molmo(raw: str) -> tuple[float, float] | None:
    match = MOLMO_COORDS.search(raw)
    if not match:
        return None
    values = match.group(1).split()
    # Triples "index x y ...", optionally after a 1-based image index (vLLM returns "1 1 770 483"): first point.
    start = 1 if len(values) % 3 == 1 else 0
    if len(values) < start + 3:
        return None
    return float(values[start + 1]), float(values[start + 2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True, help="model name as served by the endpoint")
    parser.add_argument("--name", required=True, help="short name for the results file")
    parser.add_argument("--kind", choices=["qwen", "molmo"], default="qwen")
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--prompt", choices=["demo", "native"], default="demo", help="qwen only")
    parser.add_argument("--max-dim", type=int, default=384, help="384 = what the demo sends")
    parser.add_argument("--limit", type=int, default=0, help="first N samples only (smoke test)")
    args = parser.parse_args()

    url = normalize_vlm_url(args.url)
    samples = json.loads((DATA / "refcoco_sample.json").read_text())
    if args.limit:
        samples = samples[: args.limit]

    records = []
    for i, s in enumerate(samples):
        image = resize(Image.open(DATA / s["image"]).convert("RGB"), args.max_dim)
        if args.kind == "molmo":
            text, to_point = f"Point to the {s['expression']}.", point_molmo
        elif args.prompt == "native":
            # Qwen-VL's documented grounding request: a JSON bbox_2d box on the 0-1000 grid.
            text = f'Locate "{s["expression"]}" in the image. Output its bbox coordinates in JSON format.'
            to_point = point_qwen_native
        else:
            text, to_point = custom_prompt_template(s["expression"]), point_qwen
        started = time.perf_counter()
        try:
            raw, error = ask(url, args.model, text, image, args.no_think), None
        except requests.RequestException as exc:
            raw, error = "", str(exc)
        latency = time.perf_counter() - started

        point = to_point(raw)
        hit = False
        if point is not None:
            # 0-1000 normalized -> original image pixels, scored against the xyxy ground-truth box.
            px, py = point[0] / 1000 * s["width"], point[1] / 1000 * s["height"]
            x0, y0, x1, y1 = s["bbox_xyxy"]
            hit = x0 <= px <= x1 and y0 <= py <= y1
        records.append({"ref_id": s["ref_id"], "expression": s["expression"], "point_0_1000": point,
                        "hit": hit, "latency_s": round(latency, 3), "error": error, "raw": raw[:300]})
        print(f"[{i + 1}/{len(samples)}] {'HIT ' if hit else 'miss'} {latency:5.2f}s {s['expression'][:60]}", flush=True)

    latencies = [r["latency_s"] for r in records if r["error"] is None]
    summary = {
        "name": args.name,
        "model": args.model,
        "kind": args.kind,
        "no_think": args.no_think,
        "prompt": args.prompt if args.kind == "qwen" else "molmo",
        "max_dim": args.max_dim,
        "n": len(records),
        "accuracy": round(sum(r["hit"] for r in records) / len(records), 4),
        "no_point": sum(r["point_0_1000"] is None for r in records),
        "parsed_accuracy": round(sum(r["hit"] for r in records) / max(1, sum(r["point_0_1000"] is not None for r in records)), 4),
        "errors": sum(r["error"] is not None for r in records),
        "latency_median_s": round(statistics.median(latencies), 3) if latencies else None,
        "latency_p90_s": round(statistics.quantiles(latencies, n=10)[-1], 3) if len(latencies) >= 10 else None,
    }
    out = REPO / "eval" / "results" / f"vlm-{args.name}-{args.max_dim}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
