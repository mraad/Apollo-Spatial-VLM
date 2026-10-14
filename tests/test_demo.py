import asyncio
import base64
import io
import json
import unittest

import numpy as np
from PIL import ExifTags, Image

from demo import (
    MAX_INFER_BODY_BYTES,
    DemoRuntime,
    build_app,
    exif_fov_deg,
    float32_base64,
    parse_molmo_points,
    parse_vlm_models,
    parse_vlm_targets,
    split_target_phrases,
)


class ParseVlmTargetsTest(unittest.TestCase):
    def test_tolerates_think_fences_and_trailing_prose(self):
        raw = (
            "<think>looking</think>```json\n"
            '{"targets":[{"label":"Chair","x":320,"y":650,"confidence":0.9}]}\n```'
        )
        self.assertEqual(parse_vlm_targets(raw)[0]["label"], "chair")
        raw = '{"targets":[{"label":"door","x":1,"y":2}]}\nHope this helps! {"not": "json"'
        self.assertEqual(parse_vlm_targets(raw)[0]["label"], "door")

    def test_skips_clamps_and_colors(self):
        raw = (
            '{"targets":['
            '{"label":"floor","x":1,"y":1},'
            '{"label":"opera house","x":1500,"y":-5},'
            '{"label":"harbour bridge","x":10,"y":10},'
            '{"label":"chair 2","x":10,"y":10},'
            '{"x":5,"y":5}]}'
        )
        targets = parse_vlm_targets(raw)
        self.assertEqual([t["label"] for t in targets], ["opera house", "harbour bridge", "chair 2"])
        self.assertEqual((targets[0]["x"], targets[0]["y"]), (1000, 0))
        # Unknown labels get distinct palette colors; known labels keep their fixed color.
        self.assertNotEqual(targets[0]["color"], targets[1]["color"])
        self.assertEqual(targets[2]["color"], "#ff4444")

    def test_malformed_output_means_no_targets(self):
        self.assertEqual(parse_vlm_targets('{"targets":[{"label":"red sauce","x":880,{'), [])
        self.assertEqual(parse_vlm_targets("no json here"), [])
        self.assertEqual(parse_vlm_targets('["not", "an", "object"]'), [])
        raw = '{"targets":[{"label":"cup","x":"left","y":5},"junk",{"label":"bowl","x":10,"y":20}]}'
        self.assertEqual([t["label"] for t in parse_vlm_targets(raw)], ["bowl"])


class Float32Base64Test(unittest.TestCase):
    def test_round_trips_little_endian_float32(self):
        points = np.arange(6, dtype=np.float64).reshape(2, 3) / 3
        decoded = np.frombuffer(base64.b64decode(float32_base64(points)), dtype="<f4")
        np.testing.assert_allclose(decoded, points.ravel(), rtol=1e-6)


class InferBodyLimitTest(unittest.TestCase):
    def status(self, headers):
        # Drive the ASGI app directly: the limit must answer before the body is read or the model is touched.
        app = build_app(runtime=None)
        sent = []

        async def receive():
            raise AssertionError("body was read")

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
            "scheme": "http", "path": "/api/infer", "raw_path": b"/api/infer", "query_string": b"",
            "root_path": "", "server": ("test", 80), "client": ("test", 1),
            "headers": [(k.encode(), v.encode("latin-1")) for k, v in headers.items()],
        }
        asyncio.run(app(scope, receive, send))
        return sent[0]["status"]

    def test_rejects_oversized_and_unsized_bodies_before_reading(self):
        self.assertEqual(self.status({"content-length": str(MAX_INFER_BODY_BYTES + 1)}), 413)
        self.assertEqual(self.status({"content-length": "abc"}), 413)
        self.assertEqual(self.status({"content-length": "\u00b2"}), 413)
        self.assertEqual(self.status({"transfer-encoding": "chunked"}), 411)


def jpeg_with_exif(size: tuple[int, int], **tags) -> Image.Image:
    exif = Image.Exif()
    exif.get_ifd(ExifTags.IFD.Exif).update({getattr(ExifTags.Base, k): v for k, v in tags.items()})
    buffer = io.BytesIO()
    Image.new("RGB", size).save(buffer, "JPEG", exif=exif)
    return Image.open(io.BytesIO(buffer.getvalue()))


class ExifFovTest(unittest.TestCase):
    def test_35mm_equivalent_gives_long_side_fov(self):
        # iPhone 13 main camera: 26 mm equivalent, 4:3 -> 2 * atan(200 / (26 * 500 / 43.27)) = 67.3 deg.
        image = jpeg_with_exif((300, 400), FocalLengthIn35mmFilm=26, ExifImageWidth=4032, ExifImageHeight=3024)
        fov = exif_fov_deg(image)
        assert fov is not None
        self.assertAlmostEqual(fov, 67.30, places=1)

    def test_falls_back_without_focal_length_or_when_cropped(self):
        self.assertIsNone(exif_fov_deg(jpeg_with_exif((300, 400))))
        self.assertIsNone(exif_fov_deg(jpeg_with_exif((300, 400), FocalLengthIn35mmFilm=0)))
        cropped = jpeg_with_exif((400, 400), FocalLengthIn35mmFilm=26, ExifImageWidth=4032, ExifImageHeight=3024)
        self.assertIsNone(exif_fov_deg(cropped))


class ModelChoiceTest(unittest.TestCase):
    def test_vlm_models_from_env_or_single_default(self):
        models = parse_vlm_models("a=http://h:1/v1, molmo2-8b=http://h:2/v1", "u", "m")
        self.assertEqual(models, {"a": "http://h:1/v1/chat/completions", "molmo2-8b": "http://h:2/v1/chat/completions"})
        self.assertEqual(parse_vlm_models(None, "http://h:3/v1", "m"), {"m": "http://h:3/v1/chat/completions"})
        with self.assertRaises(ValueError):
            parse_vlm_models("no-url", "u", "m")

    def test_molmo_prompt_split_and_points(self):
        self.assertEqual(
            split_target_phrases("select the opera house, the harbour bridge and the red buoy"),
            ["opera house", "harbour bridge", "red buoy"],
        )
        self.assertEqual(split_target_phrases("Find a chair, a table, and the door."), ["chair", "table", "door"])
        # vLLM prefixes the image index; a long list can hit max_tokens before the closing quote.
        self.assertEqual(parse_molmo_points('<points coords="1 1 018 652 2 048 667 3 058'), [(18.0, 652.0), (48.0, 667.0)])
        self.assertEqual(parse_molmo_points('<points coords="1 770 483">x</points>'), [(770.0, 483.0)])
        self.assertEqual(parse_molmo_points("There are none."), [])

    def test_models_endpoint_lists_only_runnable_depth_models(self):
        runtime = DemoRuntime(model_dir="depth-anything/DA3METRIC-LARGE", device="cpu", process_res=504)
        self.assertEqual(runtime.depth_models, ["da3"])  # MoGe-3 needs CUDA
        with self.assertRaises(ValueError):
            DemoRuntime(model_dir="x", device="cpu", process_res=504, default_depth_model="moge3-l")
        app = build_app(runtime)
        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
            "scheme": "http", "path": "/api/models", "raw_path": b"/api/models", "query_string": b"",
            "root_path": "", "server": ("test", 80), "client": ("test", 1), "headers": [],
        }
        asyncio.run(app(scope, receive, send))
        body = json.loads(b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"))
        self.assertEqual([m["id"] for m in body["depth"]], ["da3"])
        self.assertEqual(body["default_depth"], "da3")
        self.assertEqual(len(body["vlm"]), 1)


if __name__ == "__main__":
    unittest.main()
