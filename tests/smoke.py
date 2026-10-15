"""Exercise a running demo and every advertised model: .venv/bin/python tests/smoke.py --url http://127.0.0.1:8080"""

import argparse
import base64
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo import VERSION  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    session = requests.Session()
    for path in ("/", "/index.js", "/healthz", "/api/models", "/openapi.json"):
        response = session.get(url + path, timeout=10)
        response.raise_for_status()
    health = session.get(url + "/healthz", timeout=10).json()
    assert health["model_loaded"] and health["version"] == VERSION, health
    models = session.get(url + "/api/models", timeout=10).json()
    image = (ROOT / "Depth-Anything-3/assets/examples/SOH/000.png").read_bytes()
    rows = []
    for depth in models["depth"]:
        for prompt in ("", "locate the Sydney Opera House"):
            for vlm in models["vlm"] if prompt else [models["default_vlm"]]:
                started = time.perf_counter()
                response = session.post(url + "/api/infer", files={"image": ("sample.png", image, "image/png")},
                                        data={"depth_model": depth["id"], "vlm": vlm, "prompt": prompt}, timeout=180)
                response.raise_for_status()
                result = response.json()
                meta = result["meta"]
                size = meta["processed_size"]
                grid = np.frombuffer(base64.b64decode(result["depth"]), dtype="<f2")
                assert grid.size == size["width"] * size["height"]
                assert np.isfinite(grid).all() and (grid >= 0).all() and (grid > 0).any()
                assert meta["depth_model"] == depth["id"] and meta["focal_px"] > 0
                assert meta["target_count"] == len(result["targets_3d"])
                assert bool(result["targets_2d"]) == bool(prompt), result["targets_2d"]
                assert bool(result["targets_3d"]) == bool(prompt), result["targets_3d"]
                for target in result["targets_3d"]:
                    assert all(math.isfinite(v) for v in target["position"].values())
                    assert target["position"]["z"] < 0
                fields = ["rgb", "depth_preview"]
                if prompt:
                    fields.append("annotated_preview")
                for field in fields:
                    assert result[field], f"Missing {field}"
                    with Image.open(io.BytesIO(base64.b64decode(result[field]))) as decoded:
                        decoded.load()
                        assert decoded.size == (size["width"], size["height"])
                row = {"depth": depth["id"], "vlm": vlm if prompt else None,
                       "seconds": round(time.perf_counter() - started, 3), "bytes": len(response.content),
                       "targets_3d": len(result["targets_3d"])}
                rows.append(row)
                print(json.dumps(row), flush=True)
    for data in ({"depth_model": "unknown"}, {"vlm": "unknown"}, {}):
        response = session.post(url + "/api/infer", files={"image": ("bad.png", b"bad image")}, data=data, timeout=10)
        assert response.status_code == 400, (data, response.status_code, response.text)
    print(json.dumps({"version": VERSION, "device": health["device"], "cases": len(rows), "status": "passed"}))


if __name__ == "__main__":
    main()
