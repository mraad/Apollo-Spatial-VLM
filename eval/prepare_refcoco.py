# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow", "numpy>=2"]
# ///
"""Sample RefCOCO val referring expressions and fetch only their COCO images.

Usage: uv run eval/prepare_refcoco.py [--n 200] [--seed 0]
Reads eval/data/refcoco/data/validation-*.parquet (jxu124/refcoco), writes eval/data/refcoco_sample.json.
"""

import argparse
import json
import random
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
COCO_URL = "http://images.cocodataset.org/train2014/{}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_parquet(next((DATA / "refcoco" / "data").glob("validation-*.parquet")))
    rng = random.Random(args.seed)
    rows = df.iloc[sorted(rng.sample(range(len(df)), args.n))]

    images = DATA / "coco"
    images.mkdir(exist_ok=True)
    samples = []
    for _, row in rows.iterrows():
        info = json.loads(row["raw_image_info"])
        x, y, w, h = json.loads(row["raw_anns"])["bbox"]
        x0, y0, x1, y1 = (float(v) for v in row["bbox"])
        # The parquet box is xyxy; COCO's own annotation is xywh.
        assert abs(x0 - x) < 0.5 and abs(x1 - (x + w)) < 0.5 and abs(y1 - (y + h)) < 0.5, row["ref_id"]
        path = images / info["file_name"]
        if not path.exists():
            urllib.request.urlretrieve(COCO_URL.format(info["file_name"]), path)
        samples.append({
            "ref_id": int(row["ref_id"]),
            "image": str(path.relative_to(DATA)),
            "width": int(info["width"]),
            "height": int(info["height"]),
            "bbox_xyxy": [x0, y0, x1, y1],
            "expression": rng.choice(list(row["captions"])),
        })

    out = DATA / "refcoco_sample.json"
    out.write_text(json.dumps(samples, indent=1))
    print(f"{len(samples)} samples, {len({s['image'] for s in samples})} images -> {out}")


if __name__ == "__main__":
    main()
