"""Two-view geometry experiment; no model/server imports. See docs/two-view-localization.md."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def project(points, intrinsics, world_to_camera):
    camera = np.asarray(points) @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    pixels = camera @ intrinsics.T
    return pixels[:, :2] / pixels[:, 2:3]


def triangulate(pixels0, pixels1, k0, k1, pose0, pose1, min_angle_deg=0.1, max_reprojection_px=1.0):
    """Undistorted pixel pairs -> world metres, validity, reprojection error, ray angle.

    Poses are OpenCV world-to-camera 4x4 transforms; translations must be in metres.
    The validity gate detects some bad geometry, but cannot certify a correct match.
    """
    pixels0, pixels1, k0, k1, pose0, pose1 = [
        np.asarray(a, dtype=np.float64) for a in (pixels0, pixels1, k0, k1, pose0, pose1)
    ]
    if pixels0.ndim != 2 or pixels0.shape[1] != 2 or pixels0.shape != pixels1.shape or not len(pixels0):
        raise ValueError("Expected nonempty matching Nx2 pixel arrays")
    if not all(np.isfinite(a).all() for a in (pixels0, pixels1, k0, k1, pose0, pose1)):
        raise ValueError("Inputs must be finite")
    if not (0 <= min_angle_deg < 90 and 0 < max_reprojection_px < math.inf):
        raise ValueError("Invalid geometry thresholds")
    for k, pose in ((k0, pose0), (k1, pose1)):
        if k.shape != (3, 3) or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("Expected a pinhole intrinsic matrix with positive focal lengths")
        if pose.shape != (4, 4) or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError("Expected a homogeneous world-to-camera pose")
        rotation = pose[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1):
            raise ValueError("Pose rotation must be orthonormal and right-handed")
    centers = [-p[:3, :3].T @ p[:3, 3] for p in (pose0, pose1)]
    if np.linalg.norm(centers[1] - centers[0]) < 1e-9:
        raise ValueError("Zero baseline: rotation alone cannot triangulate depth")
    homogeneous = cv2.triangulatePoints(k0 @ pose0[:3], k1 @ pose1[:3], pixels0.T, pixels1.T).T
    points = np.full((len(pixels0), 3), np.nan)
    np.divide(homogeneous[:, :3], homogeneous[:, 3:4], out=points,
              where=np.abs(homogeneous[:, 3:4]) > 1e-12)
    errors, depths, rays = [], [], []
    for pixels, k, pose in ((pixels0, k0, pose0), (pixels1, k1, pose1)):
        with np.errstate(divide="ignore", invalid="ignore"):
            errors.append(np.linalg.norm(project(points, k, pose) - pixels, axis=1))
        depths.append((points @ pose[:3, :3].T + pose[:3, 3])[:, 2])
        ray = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(k).T @ pose[:3, :3]
        rays.append(ray / np.linalg.norm(ray, axis=1, keepdims=True))
    angles = np.degrees(np.arccos(np.clip(np.sum(rays[0] * rays[1], axis=1), -1, 1)))
    errors = np.maximum(*errors)
    valid = (np.isfinite(points).all(axis=1) & (depths[0] > 0) & (depths[1] > 0)
             & (np.minimum(angles, 180 - angles) >= min_angle_deg) & (errors <= max_reprojection_px))
    return points, valid, errors, angles


def synthetic_experiment():
    rng = np.random.default_rng(42)
    focal, pixel_sigma, samples = 1500.0, 0.5, 10000
    k = np.array([[focal, 0, 960], [0, focal, 540], [0, 0, 1]])
    rows = []
    for depth in (5.0, 10.0, 20.0, 50.0, 100.0):
        for baseline in (0.2, 1.0, 5.0):
            pose0, pose1 = np.eye(4), np.eye(4)
            pose1[0, 3] = -baseline
            truth = np.array([[baseline / 2, 0, depth]])
            p0 = project(truth, k, pose0) + rng.normal(0, pixel_sigma, (samples, 2))
            p1 = project(truth, k, pose1) + rng.normal(0, pixel_sigma, (samples, 2))
            points, valid, _, _ = triangulate(p0, p1, k, k, pose0, pose1)
            finite_front = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
            error = points[finite_front, 2] - depth
            rows.append({"depth_m": depth, "baseline_m": baseline,
                         "linear_sigma_z_m": depth ** 2 * math.sqrt(2) * pixel_sigma / (focal * baseline),
                         "empirical_sigma_z_m": float(np.std(error)),
                         "p95_abs_z_error_m": float(np.percentile(np.abs(error), 95)),
                         "geometry_gate_fraction": float(np.mean(valid)),
                         "finite_front_fraction": float(np.mean(finite_front))})
    # Small-error propagation, near the optical axis, for a separate 100 m / 5 m budget.
    z, b = 100.0, 5.0
    budget = {"pixel_matching_sigma_z_m": z*z*math.sqrt(2)*pixel_sigma/(focal*b),
              "relative_yaw_0.05deg_sigma_z_m": z*z*math.radians(0.05)/b,
              "baseline_0.01m_sigma_z_m": z*0.01/b,
              "global_attitude_0.1deg_lateral_sigma_m": z*math.radians(0.1)}
    return {"seed": 42, "samples_per_case": samples, "focal_px": focal,
            "independent_pixel_sigma_px": pixel_sigma, "rows": rows,
            "pose_budget_100m_range_5m_baseline": budget,
            "note": "Synthetic noise assumptions, not measured model or GNSS accuracy. Monte Carlo depth errors are before the geometry gate."}


def read_middlebury_pfm(path):
    """Middlebury disparities are raw pixels; PFM scale magnitude is ignored."""
    with path.open("rb") as stream:
        if stream.readline().strip() != b"Pf":
            raise ValueError("Expected a grayscale PFM")
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        if width <= 0 or height <= 0 or not math.isfinite(scale) or scale == 0:
            raise ValueError("Invalid PFM header")
        values = np.frombuffer(stream.read(), dtype="<f4" if scale < 0 else ">f4")
        if values.size != width * height:
            raise ValueError("PFM payload does not match its dimensions")
        return np.flipud(values.reshape(height, width))


def middlebury_experiment(directory):
    calibration = dict(line.split("=", 1) for line in (directory / "calib.txt").read_text().splitlines())
    matrices = [np.fromstring(calibration[key].strip("[]").replace(";", " "), sep=" ").reshape(3, 3)
                for key in ("cam0", "cam1")]
    images = [cv2.imread(str(directory / f"im{i}.png"), cv2.IMREAD_GRAYSCALE) for i in (0, 1)]
    if any(im is None for im in images) or images[0].shape != images[1].shape:
        raise ValueError("Missing or mismatched stereo images")
    gt = read_middlebury_pfm(directory / "disp0.pfm")
    if gt.shape != images[0].shape:
        raise ValueError("Ground truth and image dimensions differ")
    # Exact quarter-resolution sampling keeps the mapping back to GT pixels explicit.
    stride = 4
    images = [np.ascontiguousarray(im[::stride, ::stride]) for im in images]
    matrices = [np.diag([1 / stride, 1 / stride, 1]) @ k for k in matrices]
    gt = gt[::stride, ::stride] / stride
    count = int(math.ceil(float(calibration["ndisp"]) / stride / 16) * 16)
    params = dict(numDisparities=count, blockSize=5, P1=8*25, P2=32*25,
                  disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
                  speckleRange=2, preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    left = cv2.StereoSGBM_create(minDisparity=0, **params).compute(*images).astype(np.float32) / 16
    right = cv2.StereoSGBM_create(minDisparity=-count, **params).compute(*images[::-1]).astype(np.float32) / 16
    yy, xx = np.indices(left.shape)
    xr = xx - left
    right_at_left = cv2.remap(right, xr.astype(np.float32), yy.astype(np.float32), cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=-count-1)
    candidate = (left > 0) & (xr >= 0) & (xr < left.shape[1]-1)
    candidate &= (right_at_left > -count) & (np.abs(left + right_at_left) <= 1.0)
    y, x = np.nonzero(candidate)
    if not len(x):
        raise ValueError("No left-right consistent matches")
    p0 = np.column_stack((x, y)).astype(float)
    p1 = np.column_stack((x-left[y, x], y)).astype(float)
    pose0, pose1 = np.eye(4), np.eye(4)
    baseline = float(calibration["baseline"]) / 1000
    pose1[0, 3] = -baseline
    points, valid, _, _ = triangulate(p0, p1, *matrices, pose0, pose1)
    accepted = np.zeros_like(candidate)
    accepted[y[valid], x[valid]] = True
    depths = np.full(left.shape, np.nan)
    depths[y[valid], x[valid]] = points[valid, 2]
    gt_valid = np.isfinite(gt) & (gt > 0)
    doffs = float(calibration["doffs"]) / stride
    gt_depth = np.full(gt.shape, np.nan)
    gt_depth[gt_valid] = matrices[0][0, 0] * baseline / (gt[gt_valid] + doffs)
    scored = accepted & gt_valid
    if not scored.any():
        raise ValueError("No accepted matches with ground truth")
    dz = np.abs(depths[scored] - gt_depth[scored])
    factors = np.sqrt(1 + ((xx - matrices[0][0, 2]) / matrices[0][0, 0])**2
                      + ((yy - matrices[0][1, 2]) / matrices[0][1, 1])**2)
    xyz_error = dz * factors[scored]
    disparity_error = np.abs(left[scored] - gt[scored])
    return {"source": "https://vision.middlebury.edu/stereo/data/scenes2014/",
            "scene": directory.name, "method": "OpenCV StereoSGBM + left-right consistency + calibrated triangulation",
            "numpy": np.__version__, "opencv": cv2.__version__, "stride": stride,
            "processed_width": int(left.shape[1]), "processed_height": int(left.shape[0]),
            "focal_px": matrices[0][0, 0], "baseline_m": baseline, "doffs_px": doffs,
            "sgbm": params, "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(directory.iterdir()) if p.is_file()},
            "gt_valid_pixels": int(gt_valid.sum()), "scored_pixels": int(scored.sum()),
            "accepted_fraction_of_gt": float(scored.sum()/gt_valid.sum()),
            "gt_depth_p05_p95_m": np.percentile(gt_depth[gt_valid], [5, 95]).tolist(),
            "median_abs_depth_error_m": float(np.median(dz)), "p95_abs_depth_error_m": float(np.percentile(dz, 95)),
            "median_xyz_error_m": float(np.median(xyz_error)), "p95_xyz_error_m": float(np.percentile(xyz_error, 95)),
            "fraction_xyz_error_le_0.1m_on_accepted": float(np.mean(xyz_error <= 0.1)),
            "fraction_gt_localized_within_0.1m": float(np.sum(xyz_error <= 0.1)/gt_valid.sum()),
            "bad_disparity_gt1px_fraction_on_accepted": float(np.mean(disparity_error > 1)),
            "note": "GT used only for scoring, never acceptance. One calibrated static scene; accepted-pixel metrics omit rejected/occluded pixels. No learned depth model was run."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--middlebury", type=Path, help="Optional local scene directory: calib.txt, im0/1.png, disp0.pfm")
    parser.add_argument("--output", type=Path, default=ROOT / "eval/results/two-view-localization.json")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = {"synthetic": synthetic_experiment()}
    if args.middlebury:
        result["middlebury"] = middlebury_experiment(args.middlebury)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
