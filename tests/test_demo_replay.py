"""S1 bag replay + demo data import: the paths the cloud demo takes with the team's real recordings."""

import json
import shutil
import zipfile

import numpy as np
import pytest

from scripts.make_demo_sequence import generate
from src.integration import demo_data
from src.preprocessing.frame_loader import load_frame
from src.sensors import bag_replay

O = np.eye(4)
O[:3, :3] = bag_replay.R_LINK_OPTICAL


@pytest.fixture(scope="module")
def seq(tmp_path_factory):
    root = tmp_path_factory.mktemp("seq")
    dirs = generate(root / "frames", n_frames=4)
    return dirs, [load_frame(d) for d in dirs]


def test_short_topics_no_camera_info_tf_to_map(seq, tmp_path):
    dirs, frames = seq
    T_map_opt = frames[0].T_map_camera
    bag = bag_replay.write_bag(frames, tmp_path / "S2_bag", T_map_link=T_map_opt @ np.linalg.inv(O))
    src = bag_replay.BagSource(bag)
    topics = bag_replay.detect_topics(src.topics)
    assert (topics["rgb"], topics["depth"], topics["info"]) == ("/rgb", "/depth", None)
    paths, meta = bag_replay.export(bag, tmp_path / "out", max_frames=10)
    assert len(paths) == 4 and meta["stats"]["intrinsics_source"].startswith("fallback")
    assert meta["stats"]["tf_source"] == "tf: map <- camera_link"
    got = load_frame(paths[0])
    np.testing.assert_allclose(got.T_map_camera, T_map_opt, atol=1e-6)       # body frame -> optical handled
    w, h = frames[0].rgb.shape[1], frames[0].rgb.shape[0]
    assert (got.intrinsics.width, got.intrinsics.height) == (w, h)
    assert got.intrinsics.cx == pytest.approx(640.0 * w / 1280)               # fallback scaled to image size
    np.testing.assert_array_equal(got.rgb, frames[0].rgb)
    assert json.loads((tmp_path / "out" / "session_meta.json").read_text())["duration_s"] > 0


def test_camera_info_wins_over_fallback_and_every_n(seq, tmp_path):
    _, frames = seq
    bag = bag_replay.write_bag(frames, tmp_path / "b", rgb="/zed/left/image", depth="/zed/depth/image",
                          camera_info=True, lone_db3=False)
    paths, meta = bag_replay.export(bag, tmp_path / "out", max_frames=2)
    assert len(paths) == 2 and meta["every"] == 2
    assert meta["stats"]["intrinsics_source"].startswith("camera_info")
    assert load_frame(paths[0]).intrinsics.fx == pytest.approx(frames[0].intrinsics.fx)
    assert "no map/world/odom" in meta["stats"]["tf_source"]                  # no TF -> S2 config fallback


def test_frames_at_replays_the_moments_s3_looked_at(seq, tmp_path):
    _, frames = seq
    bag = bag_replay.write_bag(frames, tmp_path / "b")
    t = [f.timestamp for f in frames]
    paths, meta = bag_replay.export(bag, tmp_path / "out", at_times=[t[3] + 0.01, t[1] - 0.01], tol=0.03)
    assert meta["matched_times"] == [1, 0]
    assert load_frame(paths[0]).timestamp == pytest.approx(t[1])


def _s3_run(dirs, root, stamps):
    """Fake S3 run folder (src/open_vocab/run.py layout) from the mock frames' observations."""
    for i, (d, ts) in enumerate(zip(dirs, stamps)):
        f = root / f"frame_{i:06d}"
        shutil.copytree(d / "masks", f / "masks")
        obs = json.loads((d / "observation.json").read_text())
        obs["timestamp"] = ts
        (f / "observation.json").write_text(json.dumps(obs))


def test_zip_with_bag_and_video_relative_s3_masks(seq, tmp_path):
    dirs, frames = seq
    pkg = tmp_path / "pkg"
    bag_replay.write_bag(frames, pkg / "bags" / "S2_bag")
    t0 = frames[0].timestamp
    _s3_run(dirs, pkg / "output" / "s3_semantic" / "S2_run", [f.timestamp - t0 for f in frames])
    shutil.copy(bag_replay.FALLBACK_INTRINSICS, pkg / "camera_intrinsics.yaml")
    archive = tmp_path / "S2_package.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for p in pkg.rglob("*"):
            z.write(p, p.relative_to(pkg))
    made = demo_data.import_source(archive, tmp_path / "datasets", max_frames=10)
    assert [m.name for m in made] == ["S2_bag"]
    meta = json.loads((made[0] / "session_meta.json").read_text())
    assert meta["s3"]["matched"] == 4 and meta["s3"]["alignment"].startswith("video-relative")
    f0 = made[0] / "frames" / "frame_0000"
    obs = json.loads((f0 / "observation.json").read_text())
    assert (f0 / obs["detections"][0]["mask"]).exists()


def test_pipeline_runs_on_imported_dataset(seq, tmp_path):
    from src.integration.pipeline import Pipeline, frame_dirs

    dirs, frames = seq
    pkg = tmp_path / "pkg"
    bag_replay.write_bag(frames, pkg / "S3_bag", T_map_link=frames[0].T_map_camera @ np.linalg.inv(O))
    _s3_run(dirs, pkg / "s3_semantic" / "20261020" , [f.timestamp for f in frames])   # absolute stamps
    made = demo_data.import_source(pkg, tmp_path / "datasets")
    pipe = Pipeline(tmp_path / "run", parser="rules")
    for d in frame_dirs(made[0] / "frames"):
        pipe.process(d)
    r = pipe.ask("Go to the mug")
    assert r["reasoning_result"]["target_id"] == "mug_01" and not pipe.contract_errors


def test_masks_rescaled_when_s3_saw_another_resolution(seq, tmp_path):
    dirs, _ = seq
    obs = json.loads((dirs[0] / "observation.json").read_text())
    w, h = obs["image_size"]["width"], obs["image_size"]["height"]
    out = tmp_path / "f"
    out.mkdir()
    got = demo_data.copy_observation(dirs[0] / "observation.json", obs, out, (w * 2, h * 2))
    from PIL import Image

    m = Image.open(out / got["detections"][0]["mask"])
    assert m.size == (w * 2, h * 2)
    assert got["detections"][0]["bbox"][2] == pytest.approx(min(obs["detections"][0]["bbox"][2] * 2, w * 2), abs=0.1)


def test_zip_slip_is_rejected(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        demo_data.safe_extract(bad, tmp_path / "x")


def test_pack_then_reimport_keeps_s1_metadata_and_shrinks_depth(seq, tmp_path):
    from demo.pack import pack

    dirs, frames = seq
    src = tmp_path / "laptop"
    bag_replay.write_bag(frames, src / "S4_bag", camera_info=True,
                         T_map_link=frames[0].T_map_camera @ np.linalg.inv(O))
    _s3_run(dirs, src / "output" / "s3_semantic" / "S4_unseen", [f.timestamp - frames[0].timestamp for f in frames])
    bag_replay.write_bag(frames[:2], src / "S6_bag")                          # a bag nobody ran S3 on
    out = tmp_path / "demo.zip"
    assert sorted(pack([src], out, log=lambda *a: None)) == ["S4_bag", "S6_bag"]
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    assert "S4_bag/frames/frame_0000/depth.png" in names and not any(n.endswith(".npy") for n in names)
    made = demo_data.import_source(out, tmp_path / "cloud")
    s4 = next(m for m in made if m.name == "S4_bag")
    meta = json.loads((s4 / "session_meta.json").read_text())
    assert meta["stats"]["tf_source"] == "tf: map <- camera_link" and meta["s3"]["matched"] == 4
    a, b = load_frame(s4 / "frames" / "frame_0000"), frames[0]
    za, zb = a.depth_raw * a.intrinsics.depth_scale, b.depth_raw * b.intrinsics.depth_scale
    ok = (zb > 0) & np.isfinite(zb)
    assert np.abs(za[ok] - zb[ok]).max() <= 0.0006                           # 1 mm PNG is lossless enough


@pytest.mark.parametrize("with_metadata", [True, False])
def test_separate_uploads_bag_files_plus_s3_zip(seq, tmp_path, monkeypatch, with_metadata):
    """The sidebar's three upload slots: metadata.yaml + .db3, S3's zip, optional intrinsics."""
    import io

    from demo import runs

    dirs, frames = seq
    monkeypatch.setattr(runs, "STORE", tmp_path / "store")
    monkeypatch.setattr(runs, "DATASETS", tmp_path / "store" / "datasets")
    monkeypatch.setattr(runs, "RUNS", tmp_path / "store" / "runs")
    bag = bag_replay.write_bag(frames, tmp_path / "rec" / "S1_static_bag", camera_info=True, lone_db3=False,
                               T_map_link=frames[0].T_map_camera @ np.linalg.inv(O))
    db3 = next(bag.glob("*.db3"))                     # ROS 2 names it S1_static_bag_0.db3; metadata names the old file
    files = [("S1_static_bag_0.db3", io.BytesIO(db3.read_bytes()))]
    if with_metadata:
        files.append(("metadata.yaml", io.BytesIO((bag / "metadata.yaml").read_bytes())))
    s3 = tmp_path / "s3" / "20261016T050736_395812Z"
    _s3_run(dirs, s3, [f.timestamp - frames[0].timestamp for f in frames])
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as z:
        for p in s3.rglob("*"):
            z.write(p, p.relative_to(s3.parent))
    assert runs.bag_name([n for n, _ in files]) == "S1_static_bag"
    names = runs.import_parts("S1_static_bag", files, [("20261016T050736_395812Z.zip", zbuf)], parser="rules")
    assert names == ["S1_static_bag"]
    summary = json.loads((runs.RUNS / "S1_static_bag" / "run_summary.json").read_text())
    assert summary["n_frames"] == 4 and not summary["contract_errors"]
    assert summary["session_meta"]["s3"]["matched"] == 4
    assert summary["last_query"]["reasoning_result"]["target_id"] == "mug_01"
    assert not (runs.STORE / "_parts").exists() or not any((runs.STORE / "_parts").iterdir())
