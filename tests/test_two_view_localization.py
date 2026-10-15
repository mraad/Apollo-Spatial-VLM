"""Geometry checks that do not import torch or the live demo."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from eval.two_view_localization import project, read_middlebury_pfm, triangulate


class TwoViewTests(unittest.TestCase):
    def setUp(self):
        self.k = np.array([[1000., 0, 500], [0, 1000, 400], [0, 0, 1]])
        self.pose0, self.pose1 = np.eye(4), np.eye(4)
        self.pose1[0, 3] = -1

    def reconstruct(self, truth, **kwargs):
        p0 = project(truth, self.k, self.pose0)
        p1 = project(truth, self.k, self.pose1)
        return triangulate(p0, p1, self.k, self.k, self.pose0, self.pose1, **kwargs)

    def test_recovers_metric_points_with_rotated_second_camera(self):
        rotation = cv2.Rodrigues(np.array([0.02, -0.1, 0.03]))[0]
        self.pose1[:3, :3] = rotation
        self.pose1[:3, 3] = -rotation @ np.array([1., 0.2, 0.1])
        truth = np.array([[0.1, -0.2, 5], [2., 1, 20]])
        points, valid, errors, _ = self.reconstruct(truth)
        np.testing.assert_allclose(points, truth, atol=1e-9)
        self.assertTrue(valid.all())
        self.assertLess(errors.max(), 1e-9)

    def test_different_principal_points_preserve_metric_depth(self):
        truth = np.array([[0.2, 0.1, 10.]])
        k1 = self.k.copy()
        k1[0, 2] += 125
        points, valid, _, _ = triangulate(
            project(truth, self.k, self.pose0), project(truth, k1, self.pose1),
            self.k, k1, self.pose0, self.pose1)
        np.testing.assert_allclose(points, truth, atol=1e-9)
        self.assertTrue(valid.all())

    def test_pixel_noise_matches_small_error_depth_formula(self):
        rng = np.random.default_rng(7)
        truth = np.array([[0.5, 0., 10.]])
        p0 = project(truth, self.k, self.pose0) + rng.normal(0, 0.5, (10000, 2))
        p1 = project(truth, self.k, self.pose1) + rng.normal(0, 0.5, (10000, 2))
        points, _, _, _ = triangulate(p0, p1, self.k, self.k, self.pose0, self.pose1)
        expected_sigma = 10**2 * np.sqrt(2) * 0.5 / 1000
        self.assertAlmostEqual(float(np.std(points[:, 2])), expected_sigma, delta=0.004)

    def test_scaled_baseline_scales_reconstruction(self):
        truth = np.array([[0.2, 0.1, 10.]])
        p0, p1 = [project(truth, self.k, p) for p in (self.pose0, self.pose1)]
        self.pose1[0, 3] *= 2
        points, valid, _, _ = triangulate(p0, p1, self.k, self.k, self.pose0, self.pose1)
        np.testing.assert_allclose(points, truth * 2, atol=1e-9)
        self.assertTrue(valid.all())

    def test_rejects_zero_baseline(self):
        self.pose1 = self.pose0.copy()
        self.pose1[:3, :3] = cv2.Rodrigues(np.array([0., 0.1, 0.]))[0]
        with self.assertRaisesRegex(ValueError, "Zero baseline"):
            self.reconstruct(np.array([[0., 0, 5]]))

    def test_rejects_behind_camera_and_weak_parallax(self):
        _, valid, _, _ = self.reconstruct(np.array([[0., 0, -5], [0, 0, 10000]]))
        self.assertFalse(valid.any())

    def test_rejects_epipolar_mismatch(self):
        truth = np.array([[0., 0, 10]])
        p0 = project(truth, self.k, self.pose0)
        p1 = project(truth, self.k, self.pose1) + [0, 20]
        _, valid, errors, _ = triangulate(p0, p1, self.k, self.k, self.pose0, self.pose1)
        self.assertFalse(valid.any())
        self.assertGreater(errors[0], 1)

    def test_epipolar_consistent_wrong_match_can_pass(self):
        truth = np.array([[0., 0, 10]])
        p0 = project(truth, self.k, self.pose0)
        p1 = project(truth, self.k, self.pose1) + [50, 0]
        points, valid, _, _ = triangulate(p0, p1, self.k, self.k, self.pose0, self.pose1)
        self.assertTrue(valid.all())
        self.assertAlmostEqual(points[0, 2], 20.)

    def test_rejects_nonfinite_pixels_and_invalid_rotation(self):
        pixels = np.array([[0., 0]])
        with self.assertRaisesRegex(ValueError, "finite"):
            triangulate(pixels + np.nan, pixels, self.k, self.k, self.pose0, self.pose1)
        self.pose1[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            self.reconstruct(np.array([[0., 0, 10]]))

    def test_pfm_bottom_up_and_endianness(self):
        values = np.array([[1, 2], [3, np.inf]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.pfm"
            for dtype, scale in (("<f4", -0.003922), (">f4", 1)):
                path.write_bytes(f"Pf\n2 2\n{scale}\n".encode() + np.flipud(values).astype(dtype).tobytes())
                np.testing.assert_array_equal(read_middlebury_pfm(path), values)
            path.write_bytes(b"Pf\n2 2\n-1\nshort")
            with self.assertRaises(ValueError):
                read_middlebury_pfm(path)


if __name__ == "__main__":
    unittest.main()
