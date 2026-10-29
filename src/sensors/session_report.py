"""S1 session report: bag manifest, timing / sync quality, depth quality per object.

Roadmap §11.1: deliverables ``session_meta.json`` and ``depth_quality.csv``; checks "verify 30 FPS
stability" and "timestamp synchronization error"; P1 "invalid-depth percentage for glass / shiny /
dark surfaces".

    python -m src.sensors.session_report --bag data/bags/S1_static_bag --out reports/S1_static_bag/
    python -m src.sensors.session_report --bag <bag> --frames <exported frames with S3 output> --out <dir>

session_meta.json  topics, counts, rates, duration, measured FPS, frame gaps, RGB↔depth header-stamp
                   offset (mean / p95 / max), intrinsics and TF source.
depth_quality.csv  one row per frame and S3 detection (or per frame without S3 output): valid-depth %,
                   median distance, depth noise.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .bag_replay import BagSource, bag_info, detect_topics


def _stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def timing(bag, profile=None) -> dict:
    """Header-stamp statistics of the RGB and depth streams (camera clock)."""
    src = BagSource(bag)
    topics = detect_topics(src.topics, profile)
    stamps = {"rgb": [], "depth": []}
    has_info = bool(topics.get("info"))
    for topic, _, msg in src.messages([topics["rgb"], topics["depth"]]):
        stamps["rgb" if topic == topics["rgb"] else "depth"].append(_stamp(msg))
    out = {"topics": topics, "camera_info_recorded": has_info, "tf_topics": topics.get("tf", [])}
    for k, s in stamps.items():
        s = np.sort(np.asarray(s))
        dt = np.diff(s)
        if len(dt) == 0:
            out[k] = {"messages": int(len(s))}
            continue
        med = float(np.median(dt))
        out[k] = {"messages": int(len(s)), "fps_measured": round(1 / med, 2) if med > 0 else None,
                  "dt_median_ms": round(med * 1e3, 2), "dt_max_ms": round(float(dt.max()) * 1e3, 2),
                  "gaps_over_1.5x": int((dt > 1.5 * med).sum()), "duration_s": round(float(s[-1] - s[0]), 3)}
    r, d = np.sort(stamps["rgb"]), np.sort(stamps["depth"])
    if len(r) and len(d):
        idx = np.clip(np.searchsorted(d, r), 1, len(d) - 1)
        nearest = np.where(np.abs(d[idx] - r) < np.abs(d[idx - 1] - r), d[idx], d[idx - 1])
        off = np.abs(nearest - r) * 1e3
        out["rgb_depth_offset_ms"] = {"mean": round(float(off.mean()), 3), "p95": round(float(np.percentile(off, 95)), 3),
                                      "max": round(float(off.max()), 3), "pairs_within_20ms": round(float((off <= 20).mean()), 4)}
    target = 30.0
    fps = out.get("rgb", {}).get("fps_measured")
    out["fps_30_check"] = None if fps is None else {"target": target, "measured": fps, "ok": abs(fps - target) / target < 0.1}
    return out


def depth_quality(frames_dir) -> list:
    """Rows for depth_quality.csv from exported frame folders (with S3 observations when present)."""
    from ..fusion_3d.confidence import depth_noise
    from ..preprocessing.depth_cleaning import clean_depth, depth_to_meters
    from ..preprocessing.frame_loader import find_observation_file, load_frame, load_semantic_observations

    root = Path(frames_dir)
    dirs = [root] if (root / "rgb.png").exists() else sorted(p for p in root.iterdir() if (p / "rgb.png").exists())
    rows = []
    for d in dirs:
        fr = load_frame(d)
        depth = clean_depth(depth_to_meters(fr.depth_raw, fr.intrinsics.depth_scale), 0.1, 10.0)
        try:
            obs = load_semantic_observations(find_observation_file(d))
        except FileNotFoundError:
            obs = []
        regions = [(o.label, o.mask.astype(bool)) for o in obs if o.mask.shape == depth.shape] or \
                  [("(whole image)", np.ones(depth.shape, bool))]
        for label, m in regions:
            vals = depth[m]
            valid = vals[vals > 0]
            sigma = depth_noise(depth, m)
            rows.append({"frame": d.name, "object": label, "pixels": int(m.sum()),
                         "valid_depth_pct": round(100 * len(valid) / max(1, len(vals)), 2),
                         "median_distance_m": round(float(np.median(valid)), 4) if len(valid) else None,
                         "depth_noise_mm": None if sigma is None else round(sigma * 1e3, 2)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--frames", default=None, help="exported frames (bag_replay --out), optional with S3 output")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {**bag_info(BagSource(a.bag)), "timing": timing(a.bag, a.profile)}
    (out / "session_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta["timing"].get(k) for k in ("rgb", "depth", "rgb_depth_offset_ms", "fps_30_check")}, indent=2))
    if a.frames:
        rows = depth_quality(a.frames)
        with (out / "depth_quality.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} depth-quality rows")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
