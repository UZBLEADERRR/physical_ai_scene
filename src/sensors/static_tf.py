"""S1 static TF: measure map -> camera_link from the desk itself (replaces the placeholder TF).

Roadmap §11.1: "publish documented TF"; required check: "measured camera height and pitch";
deliverable ``static_tf.yaml``. On the real frames the configured TF (0.80 m, 28.6° pitch) was a
placeholder, while the desk plane put the camera at ≈ 0.32 m and ≈ 52°. This module turns that
observation into the TF.

Method, per frame: RANSAC the dominant plane (the desk) in the depth cloud (optical frame). Its
normal n (towards the camera) is the map z axis; the camera's viewing direction projected onto the
desk is the map x axis; the camera height is the plane offset d. The desk then is z = 0 in the map
frame, as S2 and S4 assume. x/y of the camera and its yaw cannot be seen from a plane: they come
from ``--xy`` / ``--yaw`` (default 0, i.e. the map origin is under the camera, x looking forward).
Several frames are combined with the median.

    python -m src.sensors.static_tf --frames outputs/pipeline/frames --out configs/sensors/
    python main_pipeline.py --bag <bag> --camera-tf configs/sensors/static_tf.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ..fusion_3d.geometry import fit_support_plane
from ..fusion_3d.projection import deproject_depth
from ..fusion_3d.transforms import R_LINK_OPTICAL, make_transform
from ..preprocessing.depth_cleaning import clean_depth, depth_to_meters


def plane_in_frame(frame, stride: int = 4, max_m: float = 3.0, threshold_m: float = 0.006):
    depth = clean_depth(depth_to_meters(frame.depth_raw, frame.intrinsics.depth_scale), 0.1, max_m)
    pts, _ = deproject_depth(depth, frame.intrinsics, stride=stride)
    return fit_support_plane(pts, threshold_m)


def T_map_optical_from_plane(plane, xy=(0.0, 0.0), yaw: float = 0.0) -> np.ndarray:
    n, d = np.asarray(plane[:3], float), float(plane[3])
    n = n / np.linalg.norm(n)
    fwd = np.array([0.0, 0.0, 1.0])                 # optical z = viewing direction
    x = fwd - (fwd @ n) * n
    if np.linalg.norm(x) < 1e-6:                    # looking straight down: use the image "up" (−y)
        x = np.array([0.0, -1.0, 0.0]) - (np.array([0.0, -1.0, 0.0]) @ n) * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    R = np.vstack([x, y, n])                        # rows: map axes in optical coordinates -> p_map = R p_opt
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R
    return make_transform(R, [xy[0], xy[1], d])


def rpy_from_matrix(R) -> list:
    """Inverse of transforms.rpy_to_matrix (Rz·Ry·Rx)."""
    pitch = float(np.arcsin(-np.clip(R[2, 0], -1, 1)))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return [roll, pitch, yaw]


def estimate(frames, xy=(0.0, 0.0), yaw: float = 0.0) -> dict:
    planes = [p for p in (plane_in_frame(f) for f in frames) if p is not None]
    if not planes:
        raise ValueError("no desk plane found in any frame")
    P = np.array(planes)
    P[:, :3] /= np.linalg.norm(P[:, :3], axis=1, keepdims=True)
    plane = np.median(P, axis=0)
    T_opt = T_map_optical_from_plane(plane, xy, yaw)
    T_link = T_opt @ np.linalg.inv(make_transform(R_LINK_OPTICAL, [0, 0, 0]))
    roll, pitch, yaw_ = rpy_from_matrix(T_link[:3, :3])
    n_opt = plane[:3]
    view_down = float(np.degrees(np.arcsin(np.clip(-n_opt @ [0, 0, 1.0], -1, 1))))   # angle of the view below horizontal
    heights = [float(p[3]) for p in P]
    return {
        "parent_frame": "map", "child_frame": "camera_link", "method": "desk_plane_ransac",
        "xyz": [round(float(x), 4) for x in T_link[:3, 3]], "rpy": [round(roll, 5), round(pitch, 5), round(yaw_, 5)],
        "camera_height_m": round(float(plane[3]), 4), "pitch_below_horizontal_deg": round(view_down, 2),
        "height_spread_m": round(float(np.max(heights) - np.min(heights)), 4), "n_frames": len(planes),
        "T_map_camera": [[round(float(v), 6) for v in r] for r in T_opt],
        "note": "x, y and yaw are not observable from the desk plane; set them with --xy / --yaw if the map origin is elsewhere",
    }


def compare(est: dict, configured: dict) -> dict:
    """Measured vs configured TF (e.g. tf_map_camera_link in configs/fusion_3d.yaml)."""
    return {"configured_height_m": configured["xyz"][2], "measured_height_m": est["xyz"][2],
            "configured_pitch_deg": round(float(np.degrees(configured["rpy"][1])), 2),
            "measured_pitch_deg": round(float(np.degrees(est["rpy"][1])), 2)}


def main():
    from ..fusion_3d.pipeline import load_config
    from ..preprocessing.frame_loader import load_frame

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="folder of frame_XXXX folders (bag_replay export) or one frame")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--xy", type=float, nargs=2, default=(0.0, 0.0))
    ap.add_argument("--yaw", type=float, default=0.0, help="radians")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.frames)
    dirs = [root] if (root / "rgb.png").exists() else sorted(p for p in root.iterdir() if (p / "rgb.png").exists())
    step = max(1, len(dirs) // a.max_frames)
    est = estimate([load_frame(d) for d in dirs[::step][: a.max_frames]], a.xy, a.yaw)
    est["vs_config"] = compare(est, load_config()["tf_map_camera_link"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "static_tf.yaml").write_text("# S1 measured static TF (src/sensors/static_tf.py)\n" +
                                        yaml.safe_dump({k: v for k, v in est.items() if k != "T_map_camera"}, sort_keys=False))
    (out / "static_tf.json").write_text(json.dumps({"T_map_camera": est["T_map_camera"], "source": "static_tf.py"}, indent=2))
    print(json.dumps({k: est[k] for k in ("xyz", "rpy", "camera_height_m", "pitch_below_horizontal_deg", "n_frames", "vs_config")}, indent=2))
    print(f"-> {out / 'static_tf.yaml'}  {out / 'static_tf.json'} (use with main_pipeline.py --camera-tf)")


if __name__ == "__main__":
    main()
