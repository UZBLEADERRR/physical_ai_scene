"""S1: checkerboard calibration, desk-plane static TF, session / depth-quality report."""

import json

import cv2
import numpy as np
import pytest

from src.fusion_3d.transforms import T_map_optical_from_static_tf
from src.preprocessing.frame_loader import load_frame
from src.preprocessing.mock_data import CAMERA_RPY, CAMERA_XYZ
from src.sensors import calibration, session_report, static_tf

K_TRUE = np.array([[640.0, 0, 640], [0, 640.0, 360], [0, 0, 1]])


def _views(n=12, seed=0, pattern=(9, 6), square=0.025, noise=0.1):
    rng = np.random.default_rng(seed)
    obj = calibration.board_points(pattern, square)
    views = []
    for _ in range(n):
        r = rng.uniform(-0.5, 0.5, 3)
        t = np.array([rng.uniform(-0.1, 0.0), rng.uniform(-0.08, 0.0), rng.uniform(0.5, 0.9)])
        img, _ = cv2.projectPoints(obj, r, t, K_TRUE, np.zeros(5))
        views.append(img.reshape(-1, 2) + rng.normal(0, noise, (len(obj), 2)))
    return views


def test_calibration_recovers_intrinsics(tmp_path):
    cal = calibration.calibrate(_views(), (1280, 720))
    assert cal.fx == pytest.approx(640, rel=0.01) and cal.K[0, 2] == pytest.approx(640, abs=5)
    assert cal.rms_px < 0.3 and not any(v["outlier"] for v in cal.per_view)
    paths = calibration.write_outputs(cal, tmp_path)
    import yaml
    y = yaml.safe_load(paths["intrinsics"].read_text())
    assert {"width", "height", "fx", "fy", "cx", "cy", "depth_scale"} <= set(y)
    assert paths["report"].read_text().startswith("view,rms_px,outlier")


def test_corner_detection_on_a_rendered_board():
    cols, rows, s = 10, 7, 40                         # 10 x 7 squares -> 9 x 6 inner corners
    board = np.kron((np.indices((rows, cols)).sum(0) % 2) * 255, np.ones((s, s))).astype(np.uint8)
    img = np.full((480, 640), 255, np.uint8)
    img[40:40 + rows * s, 60:60 + cols * s] = board
    c = calibration.find_corners(img, (9, 6))
    assert c is not None and c.shape == (54, 2)


def test_deprojection_round_trip_is_exact():
    depth = np.full((72, 128), 0.8, np.float32)
    depth[::7, ::5] = 0                               # holes are skipped
    r = calibration.deprojection_consistency(depth, K_TRUE / [[10], [10], [1]])
    assert r["n_pixels"] > 0 and r["max_px"] < 1e-9


def test_static_tf_from_desk_matches_the_true_camera_pose():
    fr = load_frame("data/mock/frame_000")
    est = static_tf.estimate([fr])
    assert est["camera_height_m"] == pytest.approx(CAMERA_XYZ[2], abs=0.01)
    assert est["pitch_below_horizontal_deg"] == pytest.approx(np.degrees(CAMERA_RPY[1]), abs=1.0)
    T_true = T_map_optical_from_static_tf(CAMERA_XYZ, CAMERA_RPY)
    assert np.allclose(np.array(est["T_map_camera"])[:3, :3], T_true[:3, :3], atol=0.02)
    cmp = static_tf.compare(est, {"xyz": [0, 0, 0.8], "rpy": [0, 0.5, 0]})
    assert cmp["configured_height_m"] == 0.8 and cmp["measured_height_m"] == pytest.approx(0.6, abs=0.01)


def test_session_report_on_a_bag(tmp_path):
    from scripts.make_demo_sequence import generate
    from src.sensors import bag_replay

    dirs = generate(tmp_path / "frames", n_frames=5)
    frames = [load_frame(d) for d in dirs]
    for k, f in enumerate(frames):
        f.timestamp = 100.0 + k / 30
    bag = bag_replay.write_bag(frames, tmp_path / "S1_bag", lone_db3=False)
    t = session_report.timing(bag)
    assert t["rgb"]["messages"] == 5 and t["rgb"]["fps_measured"] == pytest.approx(30, rel=0.01)
    assert t["fps_30_check"]["ok"] and t["rgb_depth_offset_ms"]["max"] == pytest.approx(3, abs=0.5)
    rows = session_report.depth_quality(tmp_path / "frames")
    glass = [r for r in rows if r["object"] == "glass"]
    laptop = [r for r in rows if r["object"] == "laptop"]
    assert glass and laptop and glass[0]["valid_depth_pct"] < laptop[0]["valid_depth_pct"]
    json.dumps(t)                                      # serialisable for session_meta.json
