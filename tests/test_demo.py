import asyncio
import base64
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
from PIL import ExifTags, Image

from demo import (
    MAX_INFER_BODY_BYTES,
    DemoRuntime,
    build_app,
    depth_preview_base64,
    estimate_focal_px,
    exif_fov_deg,
    float16_base64,
    parse_molmo_points,
    parse_vlm_models,
    parse_vlm_targets,
    split_target_phrases,
)


class ParseVlmTargetsTest(unittest.TestCase):
    def test_nonfinite_numbers_are_skipped_and_confidence_is_bounded(self):
        targets = parse_vlm_targets(json.dumps({"targets": [
            {"label": "bad x", "x": float("inf"), "y": 2},
            {"label": "bad confidence", "x": 1, "y": 2, "confidence": float("nan")},
            {"label": "cup", "x": 10**100, "y": -(10**100), "confidence": 4},
            {"label": "bowl", "x": 1, "y": 2, "confidence": -1},
        ]}))
        self.assertEqual([t["label"] for t in targets], ["cup", "bowl"])
        self.assertEqual([t["confidence"] for t in targets], [1, 0])
        self.assertEqual((targets[0]["x"], targets[0]["y"]), (1000, 0))
        self.assertEqual(parse_molmo_points('<points coords="1 NaN 2 2 3 inf 3 4 5">'), [(4, 5)])

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


class Float16Base64Test(unittest.TestCase):
    def test_round_trips_little_endian_float16_within_half_precision(self):
        depth = np.array([[0.0, 0.31, 4.82], [18.48, 540.0, 2731.5]], dtype=np.float32)
        decoded = np.frombuffer(base64.b64decode(float16_base64(depth)), dtype="<f2").astype(np.float32)
        np.testing.assert_allclose(decoded, depth.ravel(), rtol=1e-3)
        self.assertEqual(decoded[0], 0.0)  # invalid pixels stay exactly 0


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
    def test_rejects_nonfinite_focal_lengths_and_invalid_dimensions(self):
        for focal in (float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                estimate_focal_px(300, 400, focal)
            self.assertIsNone(exif_fov_deg(jpeg_with_exif((300, 400), FocalLengthIn35mmFilm=str(focal))))
        for width in (0, -1, "nan", "inf"):
            image = jpeg_with_exif((300, 400), FocalLengthIn35mmFilm=26, ExifImageWidth=width, ExifImageHeight=3024)
            self.assertIsNone(exif_fov_deg(image))

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
        # Malformed size tags must not raise (that would turn the upload into a 400).
        malformed = jpeg_with_exif((300, 400), FocalLengthIn35mmFilm=26, ExifImageWidth="wide", ExifImageHeight=3024)
        self.assertIsNone(exif_fov_deg(malformed))


class ModelChoiceTest(unittest.TestCase):
    def test_relative_local_model_path_is_not_sent_to_hugging_face(self):
        runtime = DemoRuntime(model_dir="./missing-model-for-test", device="cpu", process_res=504)
        with patch("demo.DepthAnything3.from_pretrained") as load:
            with self.assertRaises(FileNotFoundError):
                runtime.load_model()
            load.assert_not_called()

    def test_molmo_stops_querying_when_eight_targets_are_found(self):
        runtime = DemoRuntime(model_dir="x", device="cpu", process_res=504)
        raw = '<points coords="' + " ".join(f"{i} 200 300" for i in range(1, 11)) + '">'
        with patch.object(runtime, "ask_vlm", return_value=raw) as ask:
            targets = runtime.query_vlm_targets(Image.new("RGB", (1, 1000)), "find cups and bowls", "molmo")
        self.assertEqual(len(targets), 8)
        self.assertTrue(all(t["color"] for t in targets))
        ask.assert_called_once()
        encoded = Image.open(io.BytesIO(base64.b64decode(ask.call_args.args[2])))
        self.assertEqual(encoded.size, (1, 384))

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


class DepthOutputTest(unittest.TestCase):
    def test_preview_is_grayscale_with_invalid_pixels_black(self):
        depth = np.array([[float("nan"), float("inf"), 100], [1, 2, 3]], dtype=np.float32)
        valid = np.array([[False, False, False], [True, True, True]])
        with np.errstate(all="raise"):
            preview = Image.open(io.BytesIO(base64.b64decode(depth_preview_base64(depth, valid))))
        self.assertEqual(preview.mode, "L")
        self.assertEqual(np.asarray(preview)[0].tolist(), [0, 0, 0])

    def test_invalid_depth_is_excluded_from_targets_and_encoded_grid(self):
        runtime = DemoRuntime(model_dir="x", device="cpu", process_res=504)
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        depth = np.full((32, 32), 2.0, dtype=np.float32)
        depth[0, :3] = [float("nan"), float("inf"), 70000]
        invalid = np.zeros((32, 32), dtype=bool)
        invalid[10:23, 10:23] = True
        target = {"label": "cup", "x": 500, "y": 500, "confidence": 0.9, "color": "#ffaa44"}
        with patch.object(runtime, "predict_depth", return_value=(rgb, depth, invalid, 28, 60, "assumed")), \
             patch.object(runtime, "query_vlm_targets", return_value=[target]):
            result = runtime.infer_image(Image.fromarray(rgb), " find a cup ")
        self.assertEqual(result["targets_2d"], [target])
        self.assertEqual(result["targets_3d"], [])
        self.assertEqual(result["meta"]["prompt_used"], "find a cup")
        decoded = np.frombuffer(base64.b64decode(result["depth"]), dtype="<f2").reshape(32, 32)
        self.assertTrue(np.isfinite(decoded).all())
        self.assertTrue((decoded[invalid] == 0).all())
        self.assertEqual(decoded[0, :3].tolist(), [0, 0, 0])
        self.assertEqual(decoded[-1, -1], 2)


if __name__ == "__main__":
    unittest.main()
