"""S1 replay: read any G1 ROS 2 bag the way ``ros2 bag play`` would publish it, without ROS.

The demo cannot plug a ZED Mini into a cloud server, so this module stands in for
``ros2 bag play S2_bag``: it reads the recorded RGB, depth, camera_info and TF messages and
hands synchronized frames to the rest of the pipeline (same frame-folder layout as
``src/preprocessing/bag_reader.py``).

What it adds over ``bag_reader``:

* opens a bag directory (``metadata.yaml`` + ``.db3``/``.mcap``) **or a lone ``.db3`` file**;
* finds the topics itself: the ZED wrapper names, short names such as ``/rgb`` and ``/depth``,
  or any other ``sensor_msgs/Image`` pair;
* uses ``camera_info`` when it was recorded, else S1's fallback intrinsics
  (``configs/sensors/camera_intrinsics.yaml``), scaled to the image size;
* builds ``T_map_camera`` from ``/tf_static`` + ``/tf`` (map/world/odom → image frame);
* reports topic counts, rates and duration for S1's panel (``bag_info``).

    python -m src.sensors.bag_replay --bag data/bags/S2_bag --out data/frames/S2 --max-frames 40
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import yaml

from ..preprocessing.bag_reader import PROFILES, decode_image, intrinsics_from_camera_info, save_frame
from ..preprocessing.frame_loader import SynchronizedFrame
from ..preprocessing.intrinsics import CameraIntrinsics, load_intrinsics

REPO = Path(__file__).resolve().parents[2]
FALLBACK_INTRINSICS = REPO / "configs" / "sensors" / "camera_intrinsics.yaml"
IMAGE = "sensor_msgs/msg/Image"
INFO = "sensor_msgs/msg/CameraInfo"
TF = "tf2_msgs/msg/TFMessage"
ROOT_FRAMES = ("map", "world", "odom")
# optical frame (z forward, x right, y down) expressed in a ROS body frame (x forward, y left, z up)
R_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _typestore():
    from rosbags.typesys import Stores, get_typestore

    return get_typestore(Stores.ROS2_HUMBLE)


# ---------------------------------------------------------------- bag access
class BagSource:
    """Uniform read access to a rosbag2 directory or a lone ``.db3`` file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"bag not found: {self.path}")
        self._dir = self.path if self.path.is_dir() else self.path.parent
        self._sqlite = self.path.is_file() and self.path.suffix == ".db3" and not (self._dir / "metadata.yaml").exists()
        if self.path.is_dir() and not (self.path / "metadata.yaml").exists():
            db3 = sorted(self.path.glob("*.db3"))
            if not db3:
                raise ValueError(f"{self.path} has no metadata.yaml and no .db3 file")
            self.path, self._sqlite = db3[0], True
        self.store = _typestore()
        self.topics = self._read_topics()     # name -> {"type", "count"}

    @property
    def name(self) -> str:
        return self.path.stem if self.path.is_file() else self.path.name

    def _read_topics(self) -> dict:
        if self._sqlite:
            with sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True) as con:
                rows = con.execute("select t.name, t.type, count(m.id), min(m.timestamp), max(m.timestamp) "
                                   "from topics t left join messages m on m.topic_id = t.id group by t.id").fetchall()
            return {n: {"type": t, "count": c, "t0": (a or 0) * 1e-9, "t1": (b or 0) * 1e-9}
                    for n, t, c, a, b in rows}
        from rosbags.highlevel import AnyReader

        out = {}
        with AnyReader([self._dir], default_typestore=self.store) as r:
            for c in r.connections:
                rec = out.setdefault(c.topic, {"type": c.msgtype, "count": 0, "t0": r.start_time * 1e-9,
                                               "t1": r.end_time * 1e-9})
                rec["count"] += c.msgcount
        return out

    def messages(self, topics) -> Iterator[tuple[str, int, object]]:
        """(topic, bag time ns, deserialized message) in bag-time order, only for ``topics``."""
        topics = [t for t in topics if t in self.topics]
        if self._sqlite:
            with sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True) as con:
                ids = {n: (i, ty) for i, n, ty in con.execute("select id, name, type from topics")}
                by_id = {ids[t][0]: (t, ids[t][1]) for t in topics}
                q = (f"select topic_id, timestamp, data from messages where topic_id in "
                     f"({','.join('?' * len(by_id))}) order by timestamp")
                for tid, ns, data in con.execute(q, list(by_id)):
                    topic, msgtype = by_id[tid]
                    yield topic, ns, self.store.deserialize_cdr(data, msgtype)
            return
        from rosbags.highlevel import AnyReader

        with AnyReader([self._dir], default_typestore=self.store) as r:
            conns = [c for c in r.connections if c.topic in topics]
            for c, ns, raw in r.messages(connections=conns):
                yield c.topic, ns, r.deserialize(raw, c.msgtype)


# ---------------------------------------------------------------- topics
def detect_topics(topics: dict, profile: Optional[str] = None) -> dict:
    """Pick the RGB, depth, camera_info and TF topics of a bag (``{"rgb", "depth", "info", "tf"}``)."""
    for name in ([profile] if profile else []) + list(PROFILES):
        p = PROFILES[name]
        if p["rgb"] in topics and p["depth"] in topics:
            return {"rgb": p["rgb"], "depth": p["depth"], "info": p["info"] if p["info"] in topics else None,
                    "tf": [t for t in ("/tf_static", "/tf") if t in topics], "profile": name}
    images = [t for t, v in topics.items() if v["type"] == IMAGE]
    depth = [t for t in images if "depth" in t.lower() and "conf" not in t.lower()]
    rgb = [t for t in images if t not in depth and "conf" not in t.lower() and "right" not in t.lower()]
    if not depth or not rgb:
        raise ValueError(f"cannot find an RGB + depth Image topic pair in {sorted(images)}")

    def rank(t, words):
        return (-sum(w in t.lower() for w in words), len(t))

    rgb_t = sorted(rgb, key=lambda t: rank(t, ("rgb", "color", "left", "rect")))[0]
    depth_t = sorted(depth, key=lambda t: rank(t, ("registered", "aligned", "depth")))[0]
    infos = [t for t, v in topics.items() if v["type"] == INFO]
    prefix = rgb_t.rsplit("/", 1)[0]
    info_t = sorted(infos, key=lambda t: (not t.startswith(prefix), "depth" in t, len(t)))[0] if infos else None
    return {"rgb": rgb_t, "depth": depth_t, "info": info_t,
            "tf": [t for t, v in topics.items() if v["type"] == TF], "profile": "auto"}


# ---------------------------------------------------------------- TF
def _matrix(tr) -> np.ndarray:
    q, t = tr.rotation, tr.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    n = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [t.x, t.y, t.z]
    return T


@dataclass
class TfTree:
    """child frame -> (parent frame, 4x4 parent_T_child). Static transforms plus the first dynamic one
    of every edge: the G1 camera stands on a tripod, so the tree does not change during a bag."""

    edges: dict = field(default_factory=dict)

    def add(self, msg, static: bool):
        for ts in msg.transforms:
            child = ts.child_frame_id.lstrip("/")
            if static or child not in self.edges:
                self.edges[child] = (ts.header.frame_id.lstrip("/"), _matrix(ts.transform))

    def chain(self, frame: str) -> tuple[list[str], np.ndarray]:
        """Frames from ``frame`` up to the tree root, and root_T_frame."""
        frames, T = [frame], np.eye(4)
        while frames[-1] in self.edges and len(frames) < 64:
            parent, P = self.edges[frames[-1]]
            T = P @ T
            frames.append(parent)
        return frames, T

    def T_map_camera(self, image_frame: str) -> tuple[Optional[np.ndarray], str]:
        """4x4 optical -> map/world/odom, or (None, reason) when the tree does not reach such a frame."""
        frame = image_frame.lstrip("/")
        frames, T = self.chain(frame)
        roots = [f for f in frames if f in ROOT_FRAMES]
        if not roots:
            return None, f"TF chain {' <- '.join(reversed(frames))} reaches no {'/'.join(ROOT_FRAMES)} frame"
        T = self.chain_to(frame, roots[0])
        if "optical" not in frame:      # header in a body frame: rotate into the optical convention
            O = np.eye(4)
            O[:3, :3] = R_LINK_OPTICAL
            T = T @ O
        return T, f"tf: {roots[0]} <- {frame}"

    def chain_to(self, frame: str, root: str) -> np.ndarray:
        T, f = np.eye(4), frame
        while f != root:
            parent, P = self.edges[f]
            T, f = P @ T, parent
        return T


# ---------------------------------------------------------------- intrinsics
def fallback_intrinsics(width: int, height: int, path: str | Path = FALLBACK_INTRINSICS) -> CameraIntrinsics:
    """S1's fallback intrinsics (no S0 calibration in the bag), scaled to the recorded image size."""
    base = load_intrinsics(path)
    sx, sy = width / base.width, height / base.height
    return CameraIntrinsics(width, height, base.fx * sx, base.fy * sy, base.cx * sx, base.cy * sy, base.depth_scale)


# ---------------------------------------------------------------- frames
@dataclass
class ReplayStats:
    rgb: int = 0
    depth: int = 0
    synced: int = 0
    dropped: int = 0
    exported: int = 0
    intrinsics_source: str = ""
    tf_source: str = ""


def synced_pairs(source: BagSource, topics: dict, max_dt: float = 0.02, stats: Optional[ReplayStats] = None):
    """(stamp, rgb msg, depth msg, tf tree, camera_info intrinsics or None) for every pair whose header
    stamps differ by at most ``max_dt``. Messages are not decoded here."""
    stats = stats if stats is not None else ReplayStats()
    tree, intr, last = TfTree(), None, {"rgb": None, "depth": None}
    wanted = [topics["rgb"], topics["depth"]] + ([topics["info"]] if topics.get("info") else []) + topics["tf"]
    for topic, _, msg in source.messages(wanted):
        if topic in topics["tf"]:
            tree.add(msg, static=topic.endswith("_static"))
            continue
        if topic == topics.get("info"):
            if intr is None and len(msg.k) == 9 and msg.k[0] > 0:
                intr = intrinsics_from_camera_info(msg)
                stats.intrinsics_source = f"camera_info ({topic})"
            continue
        kind = "rgb" if topic == topics["rgb"] else "depth"
        setattr(stats, kind, getattr(stats, kind) + 1)
        last[kind] = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, msg)
        if last["rgb"] is None or last["depth"] is None:
            continue
        if abs(last["rgb"][0] - last["depth"][0]) > max_dt:
            older = "rgb" if last["rgb"][0] < last["depth"][0] else "depth"
            last[older] = None
            stats.dropped += 1
            continue
        (t_rgb, m_rgb), (t_dep, m_dep) = last["rgb"], last["depth"]
        last = {"rgb": None, "depth": None}
        stats.synced += 1
        yield min(t_rgb, t_dep), m_rgb, m_dep, tree, intr


class _FrameBuilder:
    """Decodes a selected pair into a SynchronizedFrame (intrinsics + TF resolved once per bag)."""

    def __init__(self, source, stats, intrinsics_path):
        self.source, self.stats, self.intrinsics_path = source, stats, intrinsics_path
        self.T_cache: dict = {}
        self.fallback = None

    def __call__(self, t, m_rgb, m_dep, tree, intr) -> SynchronizedFrame:
        rgb, depth = decode_image(m_rgb), decode_image(m_dep)
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"depth {depth.shape} is not registered to RGB {rgb.shape[:2]} "
                             "(ZED: depth_registered; RealSense: align_depth.enable:=true)")
        if intr is None:
            if self.fallback is None:
                self.fallback = fallback_intrinsics(rgb.shape[1], rgb.shape[0], self.intrinsics_path)
                self.stats.intrinsics_source = (f"fallback ({Path(self.intrinsics_path).name}, "
                                                f"scaled to {rgb.shape[1]}x{rgb.shape[0]})")
            intr = self.fallback
        frame_id = m_rgb.header.frame_id or "camera_optical_frame"
        if frame_id not in self.T_cache:
            T, why = tree.T_map_camera(frame_id)
            self.T_cache[frame_id] = T
            self.stats.tf_source = why if T is not None else f"{why} -> configs/fusion_3d.yaml tf_map_camera_link"
        scale = 1.0 if depth.dtype == np.float32 else 0.001
        return SynchronizedFrame(rgb=rgb, depth_raw=depth,
                                 intrinsics=CameraIntrinsics(**{**intr.to_dict(), "depth_scale": scale}),
                                 timestamp=t, frame_id=frame_id, T_map_camera=self.T_cache[frame_id],
                                 source=f"{self.source.name}@{t:.3f}")


def iter_frames(source: BagSource, topics: dict, every: int = 1, max_dt: float = 0.02,
                intrinsics_path: str | Path = FALLBACK_INTRINSICS,
                stats: Optional[ReplayStats] = None) -> Iterator[SynchronizedFrame]:
    """Synchronized RGB/depth frames, every ``every``-th pair (only the kept pairs are decoded)."""
    stats = stats if stats is not None else ReplayStats()
    build = _FrameBuilder(source, stats, intrinsics_path)
    for k, pair in enumerate(synced_pairs(source, topics, max_dt, stats)):
        if k % every == 0:
            yield build(*pair)


def frames_at(source: BagSource, topics: dict, times: list, tol: float, max_dt: float = 0.02,
              intrinsics_path: str | Path = FALLBACK_INTRINSICS,
              stats: Optional[ReplayStats] = None) -> Iterator[tuple[int, SynchronizedFrame]]:
    """For each wanted stamp (sorted, bag clock) the closest synchronized pair within ``tol`` seconds.

    Used when S3 already produced masks: the bag is replayed at exactly the moments S3 looked at.
    Yields (index into ``times``, frame); wanted stamps without a pair inside ``tol`` are skipped.
    """
    stats = stats if stats is not None else ReplayStats()
    build = _FrameBuilder(source, stats, intrinsics_path)
    order = sorted(range(len(times)), key=lambda i: times[i])
    i, best = 0, None                                       # best = (|dt|, pair) for order[i]
    for pair in synced_pairs(source, topics, max_dt, stats):
        t = pair[0]
        while i < len(order) and t > times[order[i]] + tol:
            if best is not None:
                yield order[i], build(*best[1])
            i, best = i + 1, None
        if i >= len(order):
            break
        dt = abs(t - times[order[i]])
        if dt <= tol and (best is None or dt < best[0]):
            best = (dt, pair)
    if i < len(order) and best is not None:
        yield order[i], build(*best[1])


def bag_info(source: BagSource, topics: Optional[dict] = None) -> dict:
    """Topic table + duration for S1's panel and session_meta.json."""
    t0 = min((v["t0"] for v in source.topics.values() if v["count"]), default=0.0)
    t1 = max((v["t1"] for v in source.topics.values() if v["count"]), default=0.0)
    dur = max(t1 - t0, 0.0)
    rows = [{"topic": t, "type": v["type"], "messages": v["count"],
             "rate_hz": round(v["count"] / dur, 2) if dur else None} for t, v in sorted(source.topics.items())]
    return {"bag": source.name, "duration_s": round(dur, 3), "start_time": t0, "topics": rows,
            "selected_topics": topics or {}}


def auto_every(source: BagSource, topics: dict, max_frames: int) -> int:
    n = min(source.topics[topics["rgb"]]["count"], source.topics[topics["depth"]]["count"])
    return max(1, int(np.ceil(n / max(max_frames, 1))))


def export(bag, out_dir, max_frames: int = 40, every: Optional[int] = None, profile: Optional[str] = None,
           intrinsics_path: str | Path = FALLBACK_INTRINSICS, max_dt: float = 0.02, at_times: Optional[list] = None,
           tol: float = 0.05, progress=None) -> tuple[list[Path], dict]:
    """Bag -> ``out_dir/frame_XXXX`` folders + ``out_dir/session_meta.json``. Returns (paths, session_meta).

    ``at_times`` (bag-clock seconds) replays the bag at those moments instead of every N-th pair;
    session_meta["matched_times"] then lists, per exported frame, the index of the stamp it matched.
    """
    source = BagSource(bag)
    topics = detect_topics(source.topics, profile)
    stats, paths, matched = ReplayStats(), [], []
    out_dir = Path(out_dir)
    if at_times is not None:
        frames = frames_at(source, topics, list(at_times), tol, max_dt, intrinsics_path, stats)
        expected, every = len(at_times), None
    else:
        every = every or auto_every(source, topics, max_frames)
        expected = max(1, min(max_frames, -(-auto_every(source, topics, 1) // every)))
        frames = ((None, f) for f in iter_frames(source, topics, every, max_dt, intrinsics_path, stats))
    for idx, fr in frames:
        if at_times is None and len(paths) >= max_frames:
            break
        paths.append(save_frame(fr, out_dir / f"frame_{len(paths):04d}"))
        matched.append(idx)
        if progress:
            progress(len(paths), expected)
    if not paths:
        raise ValueError(f"no synchronized RGB/depth pair in {bag} (topics {topics}, stats {asdict(stats)})")
    stats.exported = len(paths)
    meta = {**bag_info(source, topics), "every": every, "max_dt_s": max_dt, "stats": asdict(stats),
            "frames": [p.name for p in paths]}
    if at_times is not None:
        meta["matched_times"] = matched
    meta["intrinsics"] = yaml.safe_load((paths[0] / "camera_intrinsics.yaml").read_text(encoding="utf-8"))
    meta["T_map_camera"] = json.loads((paths[0] / "frame_meta.json").read_text(encoding="utf-8")).get("T_map_camera")
    (out_dir / "session_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths, meta


def rgb_time_range(bag, profile: Optional[str] = None) -> tuple[float, float]:
    """Bag-clock time of the first and last RGB message (to align S3's video-relative stamps)."""
    source = BagSource(bag)
    rec = source.topics[detect_topics(source.topics, profile)["rgb"]]
    return rec["t0"], rec["t1"]


# ---------------------------------------------------------------- writing (tests / built-in demo)
def _quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x, y, z = np.copysign(x, R[2, 1] - R[1, 2]), np.copysign(y, R[0, 2] - R[2, 0]), np.copysign(z, R[1, 0] - R[0, 1])
    return x, y, z, w


def write_bag(frames, path, rgb="/rgb", depth="/depth", camera_info=False, T_map_link=None,
                    link="camera_link", lone_db3=True):
    """Write frames as a ROS 2 bag with the short topic names S1 used (/rgb, /depth, /tf_static).

    ``camera_info=False`` reproduces a bag recorded without the S0 calibration; ``T_map_link`` adds a
    static map -> ``link`` transform; ``lone_db3`` leaves a bare ``.db3`` without metadata.yaml.
    Used by the tests and the built-in demo recording."""
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    T = ts.types
    Img, Info, Hdr, Time = (T[n] for n in ("sensor_msgs/msg/Image", "sensor_msgs/msg/CameraInfo",
                                           "std_msgs/msg/Header", "builtin_interfaces/msg/Time"))
    TFm, TS, Tr, V3, Q = (T[n] for n in ("tf2_msgs/msg/TFMessage", "geometry_msgs/msg/TransformStamped",
                                         "geometry_msgs/msg/Transform", "geometry_msgs/msg/Vector3",
                                         "geometry_msgs/msg/Quaternion"))
    ROI = T["sensor_msgs/msg/RegionOfInterest"]

    def hdr(t, frame=link):
        return Hdr(stamp=Time(sec=int(t), nanosec=int(round((t - int(t)) * 1e9))), frame_id=frame)

    with Writer(path, version=8) as w:
        c_rgb = w.add_connection(rgb, Img.__msgtype__, typestore=ts)
        c_dep = w.add_connection(depth, Img.__msgtype__, typestore=ts)
        c_tf = w.add_connection("/tf_static", TFm.__msgtype__, typestore=ts)
        c_inf = w.add_connection("/camera_info", Info.__msgtype__, typestore=ts) if camera_info else None
        t0 = frames[0].timestamp
        if T_map_link is not None:
            x, y, z, qw = _quat(T_map_link[:3, :3])
            tr = Tr(translation=V3(x=float(T_map_link[0, 3]), y=float(T_map_link[1, 3]), z=float(T_map_link[2, 3])),
                    rotation=Q(x=float(x), y=float(y), z=float(z), w=float(qw)))
            msg = TFm(transforms=[TS(header=hdr(t0, "map"), child_frame_id=link, transform=tr)])
            w.write(c_tf, int(t0 * 1e9) - 1, ts.serialize_cdr(msg, TFm.__msgtype__))
        for fr in frames:
            t, (h, wd) = fr.timestamp, fr.rgb.shape[:2]
            if c_inf is not None:
                i = fr.intrinsics
                k = np.array([i.fx, 0, i.cx, 0, i.fy, i.cy, 0, 0, 1], dtype=np.float64)
                info = Info(header=hdr(t), height=h, width=wd, distortion_model="plumb_bob", d=np.zeros(5), k=k,
                            r=np.eye(3).ravel(), p=np.zeros(12), binning_x=0, binning_y=0,
                            roi=ROI(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False))
                w.write(c_inf, int(t * 1e9), ts.serialize_cdr(info, Info.__msgtype__))
            m_rgb = Img(header=hdr(t), height=h, width=wd, encoding="rgb8", is_bigendian=0, step=wd * 3,
                        data=fr.rgb.reshape(-1))
            d = fr.depth_raw * np.float32(0.001) if fr.depth_raw.dtype == np.uint16 else fr.depth_raw.astype(np.float32)
            m_dep = Img(header=hdr(t + 0.003), height=h, width=wd, encoding="32FC1", is_bigendian=0, step=wd * 4,
                        data=d.astype("<f4").view(np.uint8).reshape(-1))
            w.write(c_rgb, int(t * 1e9), ts.serialize_cdr(m_rgb, Img.__msgtype__))
            w.write(c_dep, int((t + 0.003) * 1e9), ts.serialize_cdr(m_dep, Img.__msgtype__))
    if lone_db3:                      # S1 shared bare .db3 files: no metadata.yaml next to them
        db3 = next(path.glob("*.db3"))
        lone = path.parent / f"{path.name}.db3"
        shutil.move(db3, lone)
        shutil.rmtree(path)
        return lone
    return path



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, help="bag directory or .db3 file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--every", type=int, default=None, help="keep every N-th pair (default: spread max-frames)")
    ap.add_argument("--profile", choices=sorted(PROFILES), default=None)
    ap.add_argument("--intrinsics", default=str(FALLBACK_INTRINSICS.relative_to(REPO)))
    a = ap.parse_args()
    paths, meta = export(a.bag, a.out, a.max_frames, a.every, a.profile, a.intrinsics)
    print(json.dumps({k: meta[k] for k in ("bag", "duration_s", "every", "stats", "selected_topics")}, indent=2))
    print(f"-> {len(paths)} frames in {a.out}")


if __name__ == "__main__":
    main()
