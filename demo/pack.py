"""Pack recordings for the cloud demo: big ROS 2 bags + S3 masks -> one small zip to upload.

A 90 s HD720 ZED Mini bag is several GB; the demo only needs the frames S3 looked at. Run this on
the laptop that has the bags, then upload the zip in the demo sidebar (or share it on Google Drive
and paste the link / set DEMO_DATA_URLS).

    python -m demo.pack --src data/bags/S2_bag.db3 --src output/s3_semantic/S2_run --out S2_demo.zip
    python -m demo.pack --src recordings/ --max-frames 30 --out G1_all_bags.zip    # every bag in a folder

Depth is stored as 16-bit millimetre PNG (lossless at 1 mm, ~4x smaller than float .npy).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.integration import demo_data  # noqa: E402


def compact_depth(frame_dir: Path):
    npy = frame_dir / "depth.npy"
    if not npy.exists():
        return
    z = np.load(npy).astype(np.float32)
    mm = np.where(np.isfinite(z) & (z > 0), np.round(z * 1000), 0).clip(0, 65535).astype(np.uint16)
    Image.fromarray(mm).save(frame_dir / "depth.png")
    npy.unlink()
    intr_p = frame_dir / "camera_intrinsics.yaml"
    intr = yaml.safe_load(intr_p.read_text(encoding="utf-8"))
    intr["depth_scale"] = 0.001
    intr_p.write_text(yaml.safe_dump(intr, sort_keys=False), encoding="utf-8")


def pack(sources: list, out: Path, max_frames: int = 40, log=print) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stage = tmp / "in"
        stage.mkdir()
        for s in map(Path, sources):        # gather everything in one folder so bags and S3 runs pair up
            dst = stage / s.name
            if s.is_dir():
                shutil.copytree(s, dst)
            elif s.suffix == ".zip":
                demo_data.safe_extract(s, stage / s.stem)
            else:
                shutil.copy2(s, dst)
        made = demo_data.import_source(stage, tmp / "datasets", max_frames=max_frames,
                                       progress=lambda m, f: log(f"[{f:4.0%}] {m}"))
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for d in made:
                for f in sorted((d / "frames").iterdir()):
                    if f.is_dir():
                        compact_depth(f)
                for p in sorted(d.rglob("*")):
                    if p.is_file():
                        z.write(p, p.relative_to(d.parent))
        log(f"-> {out} ({out.stat().st_size / 2 ** 20:.1f} MB): {', '.join(m.name for m in made)}")
        return [m.name for m in made]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", required=True,
                    help="bag folder / .db3 / S3 output folder / folder with several of them / .zip (repeatable)")
    ap.add_argument("--out", required=True, help="zip to write")
    ap.add_argument("--max-frames", type=int, default=40)
    a = ap.parse_args()
    pack(a.src, Path(a.out), a.max_frames)


if __name__ == "__main__":
    main()
