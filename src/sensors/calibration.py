"""S1 camera calibration: checkerboard intrinsics + reprojection report (the skipped S0 bag).

Roadmap §11.1 deliverables: ``camera_intrinsics.yaml`` and ``calibration_report.csv``; required check:
reprojection / deprojection consistency.

    # images of a printed checkerboard (inner corners 9 x 6, 25 mm squares), 15-30 views
    python -m src.sensors.calibration --images calib/ --pattern 9x6 --square 0.025 --out configs/sensors/
    # or straight from the S0 bag (every 15th RGB frame)
    python -m src.sensors.calibration --bag data/bags/S0_calibration --every 15 --pattern 9x6 --square 0.025 --out calib_out/

The YAML has the same keys as ``configs/sensors/camera_intrinsics.yaml`` (plus the distortion), so the
bag replay and S2 can use it directly. A view whose RMS error is more than 3x the median is reported
as an outlier; recalibrate without it if there are many.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)


def board_points(pattern=(9, 6), square_m=0.025) -> np.ndarray:
    """3D corner positions on the board (z = 0), in metres, row-major like OpenCV's detector."""
    cols, rows = pattern
    g = np.zeros((rows * cols, 3), np.float32)
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return g


def find_corners(image: np.ndarray, pattern=(9, 6)):
    """Sub-pixel inner corners (N x 2) or None. ``image``: RGB/BGR or grey uint8."""
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ok, corners = cv2.findChessboardCorners(grey, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    corners = cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), CRITERIA)
    return corners.reshape(-1, 2)


@dataclass
class Calibration:
    width: int
    height: int
    K: np.ndarray
    dist: np.ndarray
    rms_px: float
    per_view: list = field(default_factory=list)     # [{"view", "rms_px", "outlier"}]

    @property
    def fx(self):
        return float(self.K[0, 0])

    def to_yaml_dict(self, source="checkerboard") -> dict:
        return {"width": self.width, "height": self.height, "fx": round(self.fx, 3), "fy": round(float(self.K[1, 1]), 3),
                "cx": round(float(self.K[0, 2]), 3), "cy": round(float(self.K[1, 2]), 3), "depth_scale": 1.0,
                "distortion_model": "plumb_bob", "distortion": [round(float(x), 6) for x in self.dist.ravel()],
                "reprojection_rms_px": round(self.rms_px, 4), "n_views": len(self.per_view), "source": source}


def calibrate(views: list, image_size, pattern=(9, 6), square_m=0.025, names=None) -> Calibration:
    """``views``: list of N x 2 corner arrays (one per image where the board was found)."""
    if len(views) < 3:
        raise ValueError(f"need at least 3 views with a detected board, got {len(views)}")
    obj = [board_points(pattern, square_m)] * len(views)
    img = [np.asarray(v, np.float32).reshape(-1, 1, 2) for v in views]
    w, h = image_size
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, (w, h), None, None)
    errs = []
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        errs.append(float(np.sqrt(np.mean(np.sum((proj - i) ** 2, axis=-1)))))
    med = float(np.median(errs))
    names = names or [f"view_{k:03d}" for k in range(len(views))]
    per = [{"view": n, "rms_px": round(e, 4), "outlier": e > 3 * med} for n, e in zip(names, errs)]
    return Calibration(w, h, K, dist, float(rms), per)


def deprojection_consistency(depth_m: np.ndarray, K: np.ndarray, stride: int = 8) -> dict:
    """Pixel -> 3D (X = (u − cx)·Z/fx, ...) -> pixel round trip on every valid depth pixel.

    Errors above ~1e-6 px mean the intrinsics are used inconsistently somewhere (e.g. a scaled K)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v, u = np.mgrid[0:depth_m.shape[0]:stride, 0:depth_m.shape[1]:stride]
    Z = depth_m[v, u].astype(np.float64)
    ok = np.isfinite(Z) & (Z > 0)
    X, Y = (u[ok] - cx) * Z[ok] / fx, (v[ok] - cy) * Z[ok] / fy
    u2, v2 = fx * X / Z[ok] + cx, fy * Y / Z[ok] + cy
    e = np.hypot(u2 - u[ok], v2 - v[ok])
    return {"n_pixels": int(ok.sum()), "max_px": float(e.max()) if len(e) else None, "mean_px": float(e.mean()) if len(e) else None}


def write_outputs(cal: Calibration, out_dir, source="checkerboard") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    y = out / "camera_intrinsics.yaml"
    y.write_text("# S1 calibration (src/sensors/calibration.py)\n" + yaml.safe_dump(cal.to_yaml_dict(source), sort_keys=False))
    c = out / "calibration_report.csv"
    with c.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["view", "rms_px", "outlier"])
        wr.writeheader()
        wr.writerows(cal.per_view)
    return {"intrinsics": y, "report": c}


def _images_from_dir(d):
    for p in sorted(Path(d).iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            im = cv2.imread(str(p))
            if im is not None:
                yield p.name, cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _images_from_bag(bag, every):
    from .bag_replay import BagSource, detect_topics, iter_frames

    src = BagSource(bag)
    for k, fr in enumerate(iter_frames(src, detect_topics(src.topics), every=every)):
        yield f"frame_{k:04d}", fr.rgb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--images", help="folder of checkerboard images")
    g.add_argument("--bag", help="S0 calibration bag (RGB topic)")
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--pattern", default="9x6", help="inner corners, columns x rows")
    ap.add_argument("--square", type=float, default=0.025, help="square size in metres")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pattern = tuple(int(x) for x in a.pattern.lower().split("x"))
    views, names, size = [], [], None
    for name, im in (_images_from_dir(a.images) if a.images else _images_from_bag(a.bag, a.every)):
        size = (im.shape[1], im.shape[0])
        c = find_corners(im, pattern)
        print(f"{name}: {'board found' if c is not None else 'no board'}")
        if c is not None:
            views.append(c)
            names.append(name)
    cal = calibrate(views, size, pattern, a.square, names)
    paths = write_outputs(cal, a.out, "checkerboard_bag" if a.bag else "checkerboard_images")
    print(f"RMS reprojection error {cal.rms_px:.3f} px over {len(views)} views; fx={cal.fx:.1f}")
    print(f"-> {paths['intrinsics']}  {paths['report']}")


if __name__ == "__main__":
    main()
