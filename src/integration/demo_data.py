"""Demo data import (S7): whatever the team recorded -> datasets the pipeline can replay.

Accepts a ``.zip``, a lone bag ``.db3`` or a folder, and finds inside it:

* ROS 2 bags (folders with ``metadata.yaml``, or lone ``.db3`` files) -- S1;
* S3 outputs: ``observation.json`` + ``masks/*.png`` (one ``frame_XXXXXX`` folder per frame, as
  written by ``src/open_vocab/run.py``) -- S3;
* ready frame folders (``rgb.png`` + depth + ``camera_intrinsics.yaml`` [+ ``observation.json``]);
* a ``camera_intrinsics.yaml`` outside any frame folder -- S1's fallback intrinsics.

Each bag becomes one dataset ``<root>/<name>/frames/frame_XXXX`` (+ ``session_meta.json``). When
S3 output belongs to the bag (same name or scenario tag, e.g. ``S2``), the bag is replayed at the
moments S3 looked at and every frame gets its ``observation.json`` and masks.

    python -m src.integration.demo_data --src S2_package.zip --out data/demo_datasets
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from ..sensors import bag_replay

Progress = Optional[Callable[[str, float], None]]


# ---------------------------------------------------------------- unpack
def safe_extract(archive: Path, dest: Path) -> Path:
    """Extract a zip without letting entries escape ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (dest / info.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"unsafe path in archive: {info.filename}")
            if "__MACOSX" in Path(info.filename).parts:
                continue
            z.extract(info, dest)
    return dest


# ---------------------------------------------------------------- discover
def _is_frame_dir(p: Path) -> bool:
    return (p / "rgb.png").exists() and any((p / n).exists() for n in ("depth.npy", "depth.png"))


def discover(root: Path) -> dict:
    root = Path(root)
    bags, frame_dirs, obs, intr = [], [], [], []
    bag_dirs = {p.parent for p in root.rglob("metadata.yaml")}
    bags += sorted(bag_dirs)
    bags += sorted(p for p in root.rglob("*.db3") if p.parent not in bag_dirs)
    if root.is_file() and root.suffix == ".db3":
        bags = [root]
    for p in sorted(root.rglob("rgb.png")):
        if _is_frame_dir(p.parent):
            frame_dirs.append(p.parent)
    frame_set = set(frame_dirs)
    obs = sorted(p for p in root.rglob("observation.json") if p.parent not in frame_set)
    intr = sorted(p for p in root.rglob("camera_intrinsics.yaml") if p.parent not in frame_set)
    runs: dict = {}
    for o in obs:   # S3 run = parent of frame_XXXXXX folders, or the folder itself for one frame
        run = o.parent.parent if re.fullmatch(r"frame_\d+", o.parent.name) else o.parent
        runs.setdefault(run, []).append(o)
    sets: dict = {}
    for d in frame_dirs:
        sets.setdefault(d.parent if d.parent != root or len(frame_dirs) > 1 else d, []).append(d)
    return {"bags": bags, "s3_runs": runs, "frame_sets": sets, "intrinsics": intr[0] if intr else None}


_TAG = re.compile(r"(?<![a-z0-9])(s\d{1,2})(?![0-9])")


def _match_text(run: Path, root: Path) -> str:
    text = run.relative_to(root).as_posix().lower() if run.is_relative_to(root) else run.name.lower()
    manifest = run / "run.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
            text += " " + " ".join(str(f.get("input_path", "")) for f in m.get("frames", [])).lower()
            text += " " + str(m.get("input", "")).lower()
        except (ValueError, AttributeError):
            pass
    return text.replace("s3_semantic", " ")


def pair_runs(bags: list, runs: dict, root: Path) -> dict:
    """bag -> S3 run directory (or None). Exact bag name first, then scenario tag (S0..S8)."""
    texts = {r: _match_text(r, root) for r in runs}
    out, used = {}, set()
    for bag in bags:
        name = (bag.stem if bag.is_file() else bag.name).lower()
        hit = [r for r, t in texts.items() if r not in used and name in t]
        if not hit:
            tag = _TAG.search(name)
            hit = [r for r, t in texts.items() if r not in used and tag and re.search(
                rf"(?<![a-z0-9]){tag.group(1)}(?![0-9])", t)]
        out[bag] = hit[0] if len(hit) == 1 else None
        if out[bag] is not None:
            used.add(out[bag])
    left_bags = [b for b in bags if out[b] is None]
    left_runs = [r for r in runs if r not in used]
    if len(left_bags) == 1 and len(left_runs) == 1:
        out[left_bags[0]] = left_runs[0]
    return out


# ---------------------------------------------------------------- S3 observations
def _obs_frames(run_obs: list) -> list[dict]:
    rows = []
    for p in sorted(run_obs):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"path": p, "data": d, "timestamp": d.get("timestamp")})
    return rows


def copy_observation(obs_path: Path, data: dict, frame_dir: Path, image_size: tuple[int, int]) -> dict:
    """Copy S3's observation.json + masks into a frame folder, rescaled if S3 saw another resolution."""
    w, h = image_size
    sw, sh = data["image_size"]["width"], data["image_size"]["height"]
    sx, sy = w / sw, h / sh
    out = json.loads(json.dumps(data))
    (frame_dir / "masks").mkdir(exist_ok=True)
    for det in out["detections"]:
        if (sx, sy) != (1.0, 1.0):
            x1, y1, x2, y2 = det["bbox"]
            det["bbox"] = [round(min(max(x1 * sx, 0), w - 1), 1), round(min(max(y1 * sy, 0), h - 1), 1),
                           round(min(max(x2 * sx, 1), w), 1), round(min(max(y2 * sy, 1), h), 1)]
        if det.get("mask_status") == "ok" and det.get("mask"):
            src = obs_path.parent / det["mask"]
            dst = frame_dir / "masks" / f"{det['detection_id']}.png"
            m = Image.open(src).convert("L")
            if m.size != (w, h):
                m = m.resize((w, h), Image.NEAREST)
            m = Image.fromarray(np.where(np.array(m) > 127, 255, 0).astype(np.uint8))
            m.save(dst)
            det["mask"] = f"masks/{dst.name}"
    out["image_size"] = {"width": w, "height": h}
    (frame_dir / "observation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _wanted_times(rows: list, bag: Path, max_frames: int) -> tuple[Optional[list], float, str]:
    """S3 stamps on the bag clock (absolute, or video-relative shifted to the bag's first RGB frame)."""
    stamps = [r["timestamp"] for r in rows]
    if not stamps or any(s is None for s in stamps):
        return None, 0.0, "S3 has no timestamps -> paired by order"
    t0, t1 = bag_replay.rgb_time_range(bag)
    absolute = t0 - 5 <= min(stamps) <= t1 + 5
    times = stamps if absolute else [t0 + s - min(stamps) for s in stamps]
    spacing = float(np.median(np.diff(sorted(times)))) if len(times) > 1 else 0.1
    return times, max(0.05, 0.5 * spacing), "absolute bag time" if absolute else "video-relative (shifted to bag start)"


# ---------------------------------------------------------------- build datasets
def _subsample(n: int, k: int) -> list[int]:
    return list(range(n)) if n <= k else sorted({int(round(i)) for i in np.linspace(0, n - 1, k)})


def build_from_bag(bag: Path, run_obs: Optional[list], out: Path, max_frames: int, intrinsics: Optional[Path],
                   progress: Progress = None) -> dict:
    frames_dir = out / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    intr = intrinsics or bag_replay.FALLBACK_INTRINSICS
    say = (lambda m, f: progress(m, f)) if progress else (lambda m, f: None)
    rows = _obs_frames(run_obs) if run_obs else []
    rows = [rows[i] for i in _subsample(len(rows), max_frames)] if rows else []
    note = "no S3 output for this bag"
    cb = lambda i, n: say(f"S1 replay {bag.name}: frame {i}/{n}", 0.1 + 0.6 * i / max(n, 1))
    if rows:
        times, tol, note = _wanted_times(rows, bag, max_frames)
        if times is not None:
            paths, meta = bag_replay.export(bag, frames_dir, intrinsics_path=intr, at_times=times, tol=tol, progress=cb)
            pairs = [(p, rows[i]) for p, i in zip(paths, meta["matched_times"])]
        else:
            paths, meta = bag_replay.export(bag, frames_dir, max_frames=len(rows), intrinsics_path=intr, progress=cb)
            pairs = list(zip(paths, rows))
        for p, r in pairs:
            rgb = Image.open(p / "rgb.png")
            copy_observation(r["path"], r["data"], p, rgb.size)
        meta["s3"] = {"run": rows[0]["path"].parent.parent.name, "alignment": note,
                      "matched": len(pairs), "s3_frames": len(rows)}
    else:
        paths, meta = bag_replay.export(bag, frames_dir, max_frames=max_frames, intrinsics_path=intr, progress=cb)
        meta["s3"] = {"run": None, "alignment": note, "matched": 0, "s3_frames": 0}
    (out / "session_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def build_from_frames(frame_dirs: list, out: Path, intrinsics: Optional[Path]) -> dict:
    frames_dir = out / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    for i, d in enumerate(sorted(frame_dirs)):
        dst = frames_dir / f"frame_{i:04d}"
        shutil.copytree(d, dst)
        if not (dst / "camera_intrinsics.yaml").exists():
            w, h = Image.open(dst / "rgb.png").size
            k = bag_replay.fallback_intrinsics(w, h, intrinsics or bag_replay.FALLBACK_INTRINSICS)
            from ..preprocessing.intrinsics import save_intrinsics

            save_intrinsics(k, dst / "camera_intrinsics.yaml")
    parent = Path(sorted(frame_dirs)[0]).parent
    old = next((m for m in (parent / "session_meta.json", parent.parent / "session_meta.json") if m.exists()), None)
    meta = json.loads(old.read_text(encoding="utf-8")) if old else {"bag": out.name, "source": "frame folders"}
    meta["frames"] = sorted(p.name for p in frames_dir.iterdir() if p.is_dir())
    meta.setdefault("s3", {})["matched"] = sum((p / "observation.json").exists() for p in frames_dir.iterdir())
    (out / "session_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "dataset"


def import_source(src: str | Path, datasets_root: str | Path, max_frames: int = 40,
                  progress: Progress = None, workdir: Optional[Path] = None) -> list[Path]:
    """Import a zip / bag / folder. Returns the dataset folders created (one per bag or frame set)."""
    src, root = Path(src), Path(datasets_root)
    root.mkdir(parents=True, exist_ok=True)
    tmp = None
    if src.suffix == ".zip":
        tmp = Path(workdir or root / "_unpack") / slug(src.stem)
        if tmp.exists():
            shutil.rmtree(tmp)
        if progress:
            progress(f"unpacking {src.name}", 0.02)
        safe_extract(src, tmp)
        scan = tmp
    else:
        scan = src
    found = discover(scan)
    pairs = pair_runs(found["bags"], found["s3_runs"], scan if scan.is_dir() else scan.parent)
    made = []
    for bag in found["bags"]:
        name = slug(bag.stem if bag.is_file() else bag.name)
        run = pairs.get(bag)
        meta = build_from_bag(bag, found["s3_runs"].get(run) if run else None, root / name, max_frames,
                              found["intrinsics"], progress)
        made.append(root / name)
        if progress:
            progress(f"{name}: {meta['stats']['exported']} frames, {meta['s3']['matched']} with S3 masks", 0.75)
    for parent, dirs in found["frame_sets"].items():
        name = slug(parent.parent.name if parent.name == "frames" else parent.name if parent != scan else src.stem)
        build_from_frames(dirs, root / name, found["intrinsics"])
        made.append(root / name)
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    if not made:
        raise ValueError(f"nothing to import in {src.name}: no ROS 2 bag (.db3 / metadata.yaml) "
                         "and no frame folders (rgb.png + depth)")
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help=".zip, bag folder, .db3 or folder of frames")
    ap.add_argument("--out", default="data/demo_datasets")
    ap.add_argument("--max-frames", type=int, default=40)
    a = ap.parse_args()
    for d in import_source(a.src, a.out, a.max_frames, progress=lambda m, f: print(f"[{f:4.0%}] {m}")):
        print("dataset:", d)


if __name__ == "__main__":
    main()
