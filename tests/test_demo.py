import asyncio
import base64
import unittest

import numpy as np

from demo import MAX_INFER_BODY_BYTES, build_app, float32_base64, parse_vlm_targets


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


if __name__ == "__main__":
    unittest.main()
