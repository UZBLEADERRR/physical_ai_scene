"""G1 demo web app: one page per team member, all fed by one pipeline run.

    streamlit run demo/app.py

The sidebar replays a recorded ZED Mini bag frame by frame (the cloud stand-in for
``ros2 bag play``); every tab shows what one module made of the current frame:

    Overview | S1 Sensors | S3 Perception | S2 3D Fusion | S6 Tracking | S4 Scene Graph |
    S5 Reasoning | S6 Action | S7 Integration

Data comes from ``demo/runs.py`` (built-in synthetic desk, uploaded zip / .db3, or a link).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo import runs  # noqa: E402
from src.fusion_3d.geometry import gravity_box_corners  # noqa: E402
from src.integration import s5_adapter  # noqa: E402
from src.integration.pipeline import load_config  # noqa: E402
from src.planning import build_action_goal, node_label, simulate  # noqa: E402

st.set_page_config(page_title="G1 Scene Understanding Demo", page_icon="🧭", layout="wide")

TEAM = {
    "S1": ("Muzaffar", "23013095", "Sensors & Data"),
    "S2": ("Sarvar", "22013120", "3D Preprocessing & Fusion"),
    "S3": ("Li Yizhe", "23013060", "Open-Vocab Perception"),
    "S4": ("Jaloliddin", "23012848", "Scene Graph Engine"),
    "S5": ("Kurbonaliyev Sanjarbek", "23012853", "Language Verifier"),
    "S6": ("Erkinov Shakhzodjon", "23013094", "Tracking & Action"),
    "S7": ("Mirzokhidbek", "23012852", "Integration & UI"),
}
PALETTE = ["#2a78d6", "#e3742b", "#3aa463", "#c64fb0", "#d6b02a", "#5bb6c9", "#8a6bd1", "#d24a4a",
           "#4f8f6f", "#b5651d"]
LABELS: list[str] = []
UPLOAD_LIMIT_MB = int(os.environ.get("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", 4000))

st.markdown("""
<style>
.g1-flow {display:flex; flex-wrap:wrap; gap:6px; align-items:stretch; margin:4px 0 14px 0;}
.g1-card {flex:1 1 128px; min-width:128px; border:1px solid rgba(128,128,128,.35); border-radius:10px;
          padding:8px 10px; background:rgba(128,128,128,.06);}
.g1-card b {font-size:0.95rem;} .g1-card .who {font-size:.78rem; opacity:.75;}
.g1-card .out {font-size:.82rem; margin-top:4px;} .g1-card .ms {font-size:.75rem; opacity:.7;}
.g1-arrow {align-self:center; opacity:.5; font-size:1.2rem;}
.g1-ok {color:#3aa463; font-weight:600;} .g1-warn {color:#e3742b; font-weight:600;}
.g1-owner {font-size:.9rem; opacity:.8; margin:-6px 0 10px 0;}
</style>""", unsafe_allow_html=True)


# ================================================================ helpers
def color_of(name: str) -> str:
    label = name.rsplit("_", 1)[0] if name[-1:].isdigit() else name
    if label not in LABELS:
        LABELS.append(label)
    return PALETTE[LABELS.index(label) % len(PALETTE)]


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def owner(sid: str):
    name, num, role = TEAM[sid]
    st.markdown(f"<div class='g1-owner'><b>{sid} · {role}</b> — {name} ({num})</div>", unsafe_allow_html=True)


def read_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


@st.cache_data(show_spinner=False)
def load_run(run_dir: str, stamp: float):
    run = Path(run_dir)
    summary = read_json(run / "run_summary.json")
    frames = []
    for f in summary["frames"]:
        frames.append({**f, "objects": read_json(run / "objects" / f"{f['frame']}_objects.json"),
                       "graph": read_json(run / "scene_graph" / f"{f['frame']}_scene_graph.json")})
    return summary, frames


@st.cache_data(show_spinner=False)
def load_rgb(path: str):
    return np.array(Image.open(path).convert("RGB"))


@st.cache_data(show_spinner=False)
def load_depth(frame_dir: str):
    d = Path(frame_dir)
    if (d / "depth.npy").exists():
        z = np.load(d / "depth.npy").astype(float)
    else:
        intr = yaml.safe_load((d / "camera_intrinsics.yaml").read_text(encoding="utf-8"))
        z = np.array(Image.open(d / "depth.png")).astype(float) * float(intr.get("depth_scale", 0.001))
    z[~np.isfinite(z) | (z <= 0)] = np.nan
    return z


@st.cache_data(show_spinner=False)
def load_points(path: str, stamp: float):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def seen_unseen():
    seen = set(read_json(ROOT / "src/open_vocab/configs/desk.json").get("prompts", []))
    unseen = {ln.strip() for ln in (ROOT / "src/open_vocab/configs/unseen_objects.txt").read_text().splitlines()
              if ln.strip() and not ln.startswith("#")}
    return seen, unseen


def image_fig(img, height=430):
    fig = go.Figure(go.Image(z=img))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height, xaxis=dict(visible=False),
                      yaxis=dict(visible=False))
    return fig


def overlay(frame, use_ids=True):
    """RGB with S3 masks/boxes; labels are S6 persistent IDs (use_ids) or S3's raw labels."""
    rgb = load_rgb(str(resolve(frame["frame_dir"]) / "rgb.png")).astype(float).copy()
    obs_path = resolve(frame["observation"])
    obs = read_json(obs_path)
    by_det = {o.get("source_observation_id"): o for o in frame["objects"]["objects"]}
    shapes, notes = [], []
    for det in obs["detections"]:
        o = by_det.get(det["detection_id"])
        pid = frame["ids"].get(o["object_id"]) if (o and use_ids) else None
        col = color_of(pid or det["labels"][0])
        if det["mask_status"] == "ok":
            m = np.array(Image.open(obs_path.parent / det["mask"]).convert("L")) > 127
            if m.shape == rgb.shape[:2]:
                rgb[m] = 0.55 * rgb[m] + 0.45 * np.array(hex_rgb(col))
        x1, y1, x2, y2 = det["bbox"]
        text = f"{pid or det['labels'][0]} {det['detector_score']:.2f}"
        if o and o.get("depth_status", "ok") != "ok" and use_ids:
            text += f" ({o['depth_status']})"
        shapes.append(dict(type="rect", x0=x1, y0=y1, x1=x2, y1=y2, line=dict(color=col, width=2)))
        notes.append(dict(x=x1, y=y1, text=text, showarrow=False, xanchor="left", yanchor="bottom",
                          font=dict(color="white", size=11), bgcolor=col))
    fig = image_fig(rgb.clip(0, 255).astype(np.uint8))
    fig.update_layout(shapes=shapes, annotations=notes)
    return fig, obs


def zone_xy(z):
    pts = z["polygon"] + [z["polygon"][0]]
    return [p[0] for p in pts], [p[1] for p in pts]


def topdown(graph, goal, height=420, tracks=None):
    fig = go.Figure()
    if tracks:
        for pid, pts in tracks.items():
            if len(pts) > 1:
                fig.add_trace(go.Scatter(x=[p[1] for p in pts], y=[p[0] for p in pts], mode="lines",
                                         line=dict(color=color_of(pid), width=1, dash="dot"), showlegend=False,
                                         hoverinfo="skip"))
    for n in graph["nodes"]:
        if n.get("virtual"):
            continue
        x, y, _ = n["pose_map"]
        fig.add_trace(go.Scatter(x=[y], y=[x], mode="markers+text", text=[n["id"]], textposition="top center",
                                 marker=dict(size=12, color="#9a9a9a" if n["state"] == "occluded" else color_of(n["id"])),
                                 showlegend=False, hovertext=f"{node_label(n)} {n['state']}"))
    if goal:
        for z in goal["exclusion_zones"]:
            xs, ys = zone_xy(z)
            fig.add_trace(go.Scatter(x=ys, y=xs, fill="toself", fillcolor="rgba(210,74,74,0.18)",
                                     line=dict(color="#d24a4a"), name=f"exclusion {z['object_id']}"))
        a = goal["agent_start"]
        fig.add_trace(go.Scatter(x=[a[1]], y=[a[0]], mode="markers", name="virtual agent",
                                 marker=dict(symbol="square", size=14, color="#555")))
        if goal["target_pose"]:
            t = goal["target_pose"]
            ok = goal["validated"]
            path = goal.get("path") or [a, t[:2]]
            fig.add_trace(go.Scatter(x=[p[1] for p in path], y=[p[0] for p in path], mode="lines",
                                     line=dict(color="#3aa463" if ok else "#d24a4a", width=3, dash="solid" if ok else "dash"),
                                     name="accepted path" if ok else "rejected path"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.2),
                      xaxis=dict(title="y (m)  ← camera right · camera left →", autorange="reversed"),
                      yaxis=dict(title="x (m) away from camera", scaleanchor="x"))
    return fig


def box_lines(box_map):
    c = gravity_box_corners(box_map)
    xs, ys, zs = [], [], []
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        xs += [c[a, 0], c[b, 0], None]
        ys += [c[a, 1], c[b, 1], None]
        zs += [c[a, 2], c[b, 2], None]
    return xs, ys, zs


def view_3d(run_dir: Path, frame, goal=None, show_cloud=True, height=520):
    fig = go.Figure()
    pts_file = run_dir / "points" / f"{frame['frame']}.npz"
    pts = load_points(str(pts_file), pts_file.stat().st_mtime) if pts_file.exists() else {}
    if show_cloud and "_scene" in pts:
        P, C = pts["_scene"], pts["_scene_rgb"]
        keep = P[:, 2] > -0.05
        if keep.sum() > 40000:
            keep &= np.random.default_rng(0).random(len(P)) < 40000 / keep.sum()
        fig.add_trace(go.Scatter3d(x=P[keep, 0], y=P[keep, 1], z=P[keep, 2], mode="markers", name="depth cloud",
                                   marker=dict(size=1.5, color=[f"rgb({r},{g},{b})" for r, g, b in C[keep]]),
                                   hoverinfo="skip"))
    for o in frame["objects"]["objects"]:
        pid = frame["ids"].get(o["object_id"], o["object_id"])
        if not show_cloud and pid in pts:
            Q = pts[pid][:: max(1, len(pts[pid]) // 3000)]
            fig.add_trace(go.Scatter3d(x=Q[:, 0], y=Q[:, 1], z=Q[:, 2], mode="markers", showlegend=False,
                                       marker=dict(size=1.6, color=color_of(pid)), hoverinfo="skip"))
        if o.get("box_map"):
            xs, ys, zs = box_lines(o["box_map"])
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color_of(pid), width=5),
                                       name=f"{pid} ({o.get('fused_confidence', 0):.2f})"))
    if goal:
        for z in goal["exclusion_zones"]:
            xs, ys = zone_xy(z)
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=[0.002] * len(xs), mode="lines", name=f"no-go {z['object_id']}",
                                       line=dict(color="#d24a4a", width=6, dash="dash")))
        if goal["target_pose"]:
            path = goal.get("path") or [goal["agent_start"], goal["target_pose"][:2]]
            fig.add_trace(go.Scatter3d(x=[p[0] for p in path], y=[p[1] for p in path], z=[0.005] * len(path), mode="lines+markers",
                                       line=dict(color="#3aa463" if goal["validated"] else "#d24a4a", width=8),
                                       name="agent path"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), legend=dict(font=dict(size=10)),
                      scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                                 camera=dict(eye=dict(x=-1.3, y=0.0, z=1.1))))
    return fig


def graph_fig(graph, height=500, show=None):
    """S4 graph drawn where the objects are on the desk (top-down); hover an edge label for its proof."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    real = [n for n in graph["nodes"] if not n.get("virtual")]
    xs = [n["pose_map"][0] for n in real] or [0.0]
    ys = [n["pose_map"][1] for n in real] or [0.0]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 0.2)

    def pos(n):        # plot x = map y (camera left on the right), plot y = map x
        if n.get("virtual"):
            return (float(np.mean(ys)), min(xs) - 0.22 * span)
        return (n["pose_map"][1], n["pose_map"][0])

    fig = go.Figure()
    pairs: dict = {}
    for e in graph["edges"]:
        if e["end_time"] is not None or e["source"] not in nodes or e["target"] not in nodes:
            continue
        if show is not None and e["predicate"] not in show:
            continue
        key = tuple(sorted((e["source"], e["target"])))
        bend = 0.06 * span * (1 + pairs.get(key, 0)) * (1 if e["source"] < e["target"] else -1)
        pairs[key] = pairs.get(key, 0) + 1
        (x0, y0), (x1, y1) = pos(nodes[e["source"]]), pos(nodes[e["target"]])
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        L = max(np.hypot(dx, dy), 1e-6)
        cx, cy = mx - dy / L * bend, my + dx / L * bend        # control point -> curved edge
        t = np.linspace(0, 1, 20)
        bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
        by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
        col = {"on": "#3aa463", "near": "#2a78d6", "left_of": "#c64fb0"}.get(e["predicate"], "#888")
        fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", line=dict(color=col, width=1.6), hoverinfo="skip",
                                 showlegend=False))
        if e["predicate"] != "near":                               # arrow head for directed relations
            fig.add_annotation(x=bx[-3], y=by[-3], ax=bx[-6], ay=by[-6], xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.4, arrowcolor=col, text="")
        ev = "<br>".join(f"{k} = {v}" for k, v in e["evidence"].items() if k != "frame")
        fig.add_trace(go.Scatter(x=[bx[10]], y=[by[10]], mode="text", text=[e["predicate"]], showlegend=False,
                                 textfont=dict(size=10, color=col),
                                 hovertext=f"<b>{e['source']} {e['predicate']} {e['target']}</b><br>{ev}"
                                           f"<br>confidence {e.get('confidence')}", hoverinfo="text"))
    for n in graph["nodes"]:
        x, y = pos(n)
        occl = n["state"] == "occluded"
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text", text=[n["id"]], textposition="top center", showlegend=False,
            marker=dict(size=26 if n.get("virtual") else 20, symbol="square" if n.get("virtual") else "circle",
                        color="#bbbbbb" if occl else color_of(n["id"]), line=dict(color="white", width=1.5)),
            hovertext=f"<b>{n['id']}</b> ({node_label(n)})<br>state: {n['state']}<br>pose: {n['pose_map']}"
                      f"<br>confidence: {n['confidence']}", hoverinfo="text"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), hovermode="closest",
                      xaxis=dict(autorange="reversed", visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def edge_key(e):
    return (e["source"], e["predicate"], e["target"])


# ================================================================ sidebar: data + replay
def progress_box():
    bar = st.progress(0.0)
    txt = st.empty()

    def cb(msg, frac):
        bar.progress(float(frac))
        txt.caption(msg)
    return cb


def imported(names):
    """Select the first new recording and report on the next run (widgets cannot change mid-run)."""
    st.session_state._select = names[0]
    st.session_state._flash = [("success", f"Imported: {', '.join(names)}")] + [("warning", e) for e in runs.LAST_ERRORS]
    st.cache_data.clear()
    st.rerun()


def sidebar_data(parser_mode):
    with st.sidebar.expander("➕ Add recording", expanded=False):
        if os.environ.get("K_SERVICE"):          # set by Google Cloud Run
            st.warning("Cloud Run accepts uploads up to 32 MB. For bags use a Google Drive link (bottom).")
        max_frames = st.slider("Max frames per bag", 5, 120, 40, 5,
                               help="The bag is replayed at S3's timestamps, or every N-th frame up to this many.")
        st.markdown("**1 · S1 bag** — `metadata.yaml` + `.db3` (select both)")
        bag = st.file_uploader("S1 bag files", type=["yaml", "db3", "mcap"], accept_multiple_files=True,
                               key="up_bag", label_visibility="collapsed")
        st.markdown("**2 · S3 output** — zip with `observation.json` + masks")
        s3 = st.file_uploader("S3 output zip", type=["zip"], accept_multiple_files=True, key="up_s3",
                              label_visibility="collapsed")
        st.markdown("**3 · Intrinsics** (optional) — `camera_intrinsics.yaml`")
        intr = st.file_uploader("camera_intrinsics.yaml", type=["yaml", "yml"], key="up_intr",
                                label_visibility="collapsed",
                                help="Only needed when the bag has no camera_info topic.")
        names = [f.name for f in bag or []]
        name = st.text_input("Recording name", value=runs.bag_name(names) if names else "",
                             placeholder="e.g. S1_static_bag")
        if bag:
            if not any(n.endswith((".db3", ".mcap")) for n in names):
                st.caption("⚠ add the .db3 file too")
            elif "metadata.yaml" not in names:
                st.caption("metadata.yaml missing: the .db3 is read directly (works for ROS 2 Humble bags).")
            if not s3:
                st.caption("⚠ no S3 output: masks are needed unless the server runs YOLO-World (WITH_S3_LIVE=1).")
        if st.button("Import recording", type="primary", width="stretch", disabled=not bag):
            try:
                out = runs.import_parts(name or runs.bag_name(names), [(f.name, f) for f in bag],
                                        [(f.name, f) for f in s3 or []], (intr.name, intr) if intr else None,
                                        parser_mode, max_frames, progress_box())
                imported(out)
            except Exception as e:
                st.error(f"Import failed: {e}")
        st.divider()
        st.caption("…or everything in **one .zip** (bags + S3 folders), or a link to such a zip.")
        up = st.file_uploader("One zip with everything", type=["zip", "db3"], key="up_zip",
                              help=f"Limit {UPLOAD_LIMIT_MB} MB.")
        if up is not None and st.button("Import zip", width="stretch"):
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / up.name
                with open(p, "wb") as f:
                    f.write(up.getbuffer())
                try:
                    imported(runs.import_and_run(p, parser_mode, max_frames, progress_box()))
                except Exception as e:
                    st.error(f"Import failed: {e}")
        url = st.text_input("Link (Google Drive / direct .zip / .db3)")
        if url and st.button("Import link", width="stretch"):
            try:
                imported(runs.import_url(url, parser_mode, max_frames, progress_box()))
            except Exception as e:
                st.error(f"Import failed: {e}")


@st.cache_data(show_spinner=False)
def code_version() -> str:
    """Short git commit of the running code (or the one Railway built), shown in the sidebar."""
    import subprocess

    sha = os.environ.get("RAILWAY_GIT_COMMIT_SHA", "")[:7]
    if not sha:
        try:
            sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                 text=True, timeout=5).stdout.strip()
        except Exception:
            sha = ""
    return sha or "unknown"


def sidebar():
    st.sidebar.title("G1 · Physical AI demo")
    st.sidebar.caption(f"code version `{code_version()}`")
    parser_mode = st.sidebar.radio("S5 parser", ["auto", "rules", "ollama"], horizontal=True,
                                   help="auto = local LLM (Ollama, OLLAMA_HOST) if reachable, else the offline "
                                        "rule parser. The verifier is the same either way.")
    parser = s5_adapter.resolve_parser(parser_mode)
    st.sidebar.caption(f"S5 parser in use: **{parser}**")
    datasets = [d for d in runs.list_datasets() if d["run"]]
    if not datasets:
        sidebar_data(parser_mode)
        return None, parser, parser_mode
    names = [d["name"] for d in datasets]
    if st.session_state.get("_select") in names:
        st.session_state.dataset = st.session_state.pop("_select")
    for kind, msg in st.session_state.pop("_flash", []):
        getattr(st.sidebar, kind)(msg)
    if st.session_state.get("dataset") not in names:
        st.session_state.dataset = names[0]
    name = st.sidebar.selectbox("Recording", names, key="dataset",
                                format_func=lambda n: f"{n} (synthetic)" if n == runs.BUILTIN else n)
    ds = next(d for d in datasets if d["name"] == name)
    sidebar_data(parser_mode)
    with st.sidebar.expander("⚙️ Manage", expanded=False):
        if st.button("Re-run pipeline on this recording", width="stretch"):
            runs.run_dataset(name, parser_mode, progress=progress_box())
            st.cache_data.clear()
            st.rerun()
        if not ds["builtin"] and st.button("Delete this recording", width="stretch"):
            runs.delete_dataset(name)
            st.session_state.pop("dataset", None)
            st.cache_data.clear()
            st.rerun()
    return ds, parser, parser_mode


def replay_controls(n):
    """Frame slider + play/pause: the stand-in for `ros2 bag play`."""
    key = f"frame_{st.session_state.dataset}"
    if key not in st.session_state:
        st.session_state[key] = n - 1
    playing = st.session_state.get("playing", False)
    st.sidebar.markdown("**▶ Bag replay** (`ros2 bag play`)")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("⏸ Pause" if playing else "▶ Play", width="stretch"):
        playing = st.session_state.playing = not playing        # handled before the frame advances
    if c2.button("⏮ Start", width="stretch"):
        st.session_state[key] = -1 if playing else 0
    if playing:
        st.session_state[key] = (st.session_state[key] + 1) % n
    speed = st.sidebar.select_slider("Seconds per frame", [0.5, 1.0, 1.5, 2.5, 4.0], value=1.0)
    k = st.sidebar.slider("Frame", 0, n - 1, key=key, disabled=playing)
    return k, playing, speed


# ================================================================ tabs
def tab_overview(summary, frames, frame, goal_state):
    st.markdown("One pipeline, seven modules, strict JSON contracts. Each card shows what that module made "
                "of the **current frame**; the module buttons above show the details.")
    objs = frame["objects"]["objects"]
    g = frame["graph"]
    active = [e for e in g["edges"] if e["end_time"] is None]
    low = [o for o in objs if o.get("depth_status", "ok") != "ok"]
    rr, goal = (goal_state or {}).get("rr"), (goal_state or {}).get("goal")
    t = frame["timings_ms"]
    meta = summary.get("session_meta", {})
    cards = [
        ("S1", f"{meta.get('stats', {}).get('exported', summary['n_frames'])} frames · "
               f"{(meta.get('intrinsics') or {}).get('width', '?')}×{(meta.get('intrinsics') or {}).get('height', '?')}",
         t.get("s1_load")),
        ("S3", f"{len(read_json(resolve(frame['observation']))['detections'])} detections + masks", t.get("s3_perception")),
        ("S2", f"{len(objs)} ObjectInstance3D" + (f" · {len(low)} low-depth" if low else ""), t.get("s2_fusion")),
        ("S6", f"{len(set(frame['ids'].values()))} persistent IDs", t.get("s6_tracking")),
        ("S4", f"{len(g['nodes'])} nodes · {len(active)} edges", t.get("s4_graph")),
        ("S5", ("verified ✓ " + (rr.get("target_id") or "")) if rr and rr["verification_status"]
         else ("refused (no proof)" if rr else "ask a question"), None),
        ("S6", ("ActionGoal ✓ validated" if goal["validated"] else "ActionGoal ✗ rejected") if goal else "—", None),
        ("S7", f"{len(summary['contract_errors'])} contract errors · {summary['fps']:.2f} FPS", None),
    ]
    html = "<div class='g1-flow'>"
    for i, (sid, out, ms) in enumerate(cards):
        name, _, role = TEAM[sid]
        label = "Action" if (sid == "S6" and i == 6) else role
        html += (f"<div class='g1-card'><b>{sid}</b> · {label}<div class='who'>{name}</div>"
                 f"<div class='out'>{out}</div>" + (f"<div class='ms'>{ms:.0f} ms</div>" if ms is not None else "")
                 + "</div>")
        if i < len(cards) - 1:
            html += "<div class='g1-arrow'>›</div>"
    st.markdown(html + "</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig, _ = overlay(frame)
        st.plotly_chart(fig, width="stretch", key="ov_cam")
    with c2:
        st.plotly_chart(topdown(g, goal, height=430), width="stretch", key="ov_top")
    m = st.columns(5)
    m[0].metric("Frames replayed", summary["n_frames"])
    m[1].metric("Latency S1→S4", f"{summary['mean_latency_ms']:.0f} ms/frame")
    m[2].metric("Throughput", f"{summary['fps']:.2f} FPS")
    m[3].metric("Contract errors", len(summary["contract_errors"]))
    m[4].metric("S5 parser", summary.get("s5_parser", "?"))
    st.markdown("**Team**")
    st.dataframe(pd.DataFrame([{"ID": k, "Member": v[0], "Student no.": v[1], "Module": v[2]} for k, v in TEAM.items()]),
                 hide_index=True, width="stretch")


def tab_s1(summary, frame):
    owner("S1")
    d = resolve(frame["frame_dir"])
    meta = summary.get("session_meta", {})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**RGB** (left rectified)")
        st.plotly_chart(image_fig(load_rgb(str(d / "rgb.png")), 380), width="stretch", key="s1_rgb")
    with c2:
        st.markdown("**Depth** (registered to RGB, metres; grey = no depth)")
        z = load_depth(str(d))
        fig = go.Figure(go.Heatmap(z=z, colorscale="Turbo", colorbar=dict(title="m"), zmin=np.nanpercentile(z, 2),
                                   zmax=np.nanpercentile(z, 98)))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=0, b=0), yaxis=dict(autorange="reversed", visible=False),
                          xaxis=dict(visible=False), plot_bgcolor="#888")
        st.plotly_chart(fig, width="stretch", key="s1_depth")
        st.caption(f"valid depth: {np.isfinite(z).mean():.1%} of pixels · median {np.nanmedian(z):.3f} m")
    intr = yaml.safe_load((d / "camera_intrinsics.yaml").read_text(encoding="utf-8"))
    fm = read_json(d / "frame_meta.json") if (d / "frame_meta.json").exists() else {}
    T = fm.get("T_map_camera") or meta.get("T_map_camera")
    stats = meta.get("stats", {})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Camera intrinsics** (`camera_intrinsics.yaml`)")
        st.dataframe(pd.DataFrame([{"param": k, "value": v} for k, v in intr.items()]), hide_index=True,
                     width="stretch")
        src = stats.get("intrinsics_source", "frame folder")
        (st.warning if src.startswith("fallback") else st.success)(f"source: {src}")
    with c2:
        st.markdown("**Extrinsics** `T_map_camera` (optical → map)")
        if T:
            T = np.array(T)
            zc = T[:3, 2]
            st.metric("camera height", f"{T[2, 3]:.3f} m")
            st.metric("camera pitch (down)", f"{np.degrees(np.arcsin(-zc[2])):.1f}°")
            st.code(np.array2string(T, precision=3, suppress_small=True), language="text")
        else:
            st.info("No TF in this recording: S2 uses `tf_map_camera_link` from configs/fusion_3d.yaml.")
        st.caption(f"TF: {stats.get('tf_source', 'frame_meta.json')}")
    with c3:
        st.markdown("**Recording**")
        st.metric("bag", meta.get("bag", summary.get("dataset", "?")))
        st.metric("duration", f"{meta.get('duration_s', 0):.1f} s")
        if stats:
            st.caption(f"RGB {stats.get('rgb')} · depth {stats.get('depth')} msgs → {stats.get('synced')} synchronized "
                       f"pairs (|Δt| ≤ {meta.get('max_dt_s', 0.02) * 1000:.0f} ms), {stats.get('dropped')} dropped, "
                       f"{stats.get('exported')} replayed")
        st.caption(f"frame time: {frame.get('timestamp', 0):.3f} s")
    if meta.get("topics"):
        st.markdown("**Topics in the bag** (what `ros2 bag play` publishes)")
        sel = set(v for v in (meta.get("selected_topics") or {}).values() if isinstance(v, str))
        sel |= set((meta.get("selected_topics") or {}).get("tf", []))
        st.dataframe(pd.DataFrame([{**r, "used": "✓" if r["topic"] in sel else ""} for r in meta["topics"]]),
                     hide_index=True, width="stretch")


def tab_s3(frame):
    owner("S3")
    fig, obs = overlay(frame, use_ids=False)
    seen, unseen = seen_unseen()
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(fig, width="stretch", key="s3_img")
        st.caption("YOLO-World v2 (text prompts → boxes) → SAM 2 (box prompt → pixel mask). Zero training.")
    with c2:
        rows = []
        p = resolve(frame["observation"])
        for det in obs["detections"]:
            area = 0
            if det["mask_status"] == "ok":
                area = int((np.array(Image.open(p.parent / det["mask"]).convert("L")) > 127).sum())
            lab = det["labels"][0]
            rows.append({"id": det["detection_id"], "label": lab, "candidates": ", ".join(det["labels"][1:]),
                         "score": round(det["detector_score"], 3), "mask px": area, "mask": det["mask_status"],
                         "split": "unseen" if lab in unseen else ("seen" if lab in seen else "open-vocab")})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, width="stretch")
        if len(df):
            st.bar_chart(df.set_index("id")["score"], height=180)
        st.caption(f"schema {obs.get('schema_version')} · image {obs['image_size']['width']}×{obs['image_size']['height']}")
    rgb = load_rgb(str(resolve(frame["frame_dir"]) / "rgb.png"))
    oks = [d for d in obs["detections"] if d["mask_status"] == "ok"]
    if oks:
        st.markdown("**Masks** (cut-outs handed to S2)")
        cols = st.columns(min(6, len(oks)))
        for i, det in enumerate(oks):
            m = np.array(Image.open(p.parent / det["mask"]).convert("L")) > 127
            x1, y1, x2, y2 = (int(v) for v in det["bbox"])
            cut = np.where(m[..., None], rgb, (rgb * 0.15).astype(np.uint8))[y1:y2, x1:x2]
            if cut.size:
                th = Image.fromarray(cut)
                th = th.resize((max(1, int(th.width * 150 / th.height)), 150))
                cols[i % len(cols)].image(th, caption=f"{det['labels'][0]} {det['detector_score']:.2f}")


def tab_s2(run_dir, frame, goal):
    owner("S2")
    objs = frame["objects"]["objects"]
    c1, c2 = st.columns([3, 2])
    with c1:
        mode = st.radio("3D view", ["Scene point cloud + boxes", "Object points only"], horizontal=True, key="s2_mode")
        st.plotly_chart(view_3d(run_dir, frame, goal, show_cloud=mode.startswith("Scene")), width="stretch",
                        key="s2_3d")
        st.caption("Mask filter → deproject X=(u−cx)Z/fx, Y=(v−cy)Z/fy → desk RANSAC, voxel, outliers, DBSCAN → "
                   "centroid + gravity-aligned box (map frame, z up).")
    with c2:
        rows = []
        for o in objs:
            bm = o.get("box_map") or {}
            rows.append({"object": frame["ids"].get(o["object_id"], o["object_id"]), "label": o["semantic_label"],
                         "x,y,z (m)": ", ".join(f"{v:.3f}" for v in (bm.get("center") or o.get("centroid_map") or o["centroid"])),
                         "W×H×D (cm)": "×".join(f"{v * 100:.1f}" for v in (bm.get("dimensions") or o["box_dimensions"])),
                         "depth conf": o.get("depth_confidence"), "depth status": o.get("depth_status", "ok"),
                         "geometry": o.get("geometry_source", "depth"), "fused conf": o.get("fused_confidence"),
                         "points": o.get("num_points")})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, width="stretch")
        if len(df):
            fig = go.Figure(go.Bar(x=df["object"], y=df["depth conf"],
                                   marker_color=["#d24a4a" if s != "ok" else "#3aa463" for s in df["depth status"]]))
            fig.update_layout(height=220, margin=dict(l=0, r=0, t=24, b=0), yaxis=dict(range=[0, 1]),
                              title=dict(text="depth confidence (red = transparent / no depth)", font=dict(size=12)))
            st.plotly_chart(fig, width="stretch", key="s2_conf")
        rej = frame["objects"].get("rejected", [])
        if rej:
            st.warning("Rejected: " + "; ".join(f"{r.get('object_id') or r.get('detection_id')}: {r.get('reason')}" for r in rej))
    with st.expander("Adaptive fusion weights per object"):
        st.dataframe(pd.DataFrame([{"object": frame["ids"].get(o["object_id"], o["object_id"]),
                                    **{f"q_{k}": v for k, v in (o.get("modality_quality") or {}).items()},
                                    **{f"w_{k}": v for k, v in (o.get("fusion_weights") or {}).items()}} for o in objs]),
                     hide_index=True, width="stretch")
    with st.expander("ObjectInstance3D JSON (hand-over to S6 / S4)"):
        st.json(frame["objects"], expanded=False)


def tracks_of(frames, upto):
    tracks = {}
    for f in frames[: upto + 1]:
        for o in f["objects"]["objects"]:
            pid = f["ids"].get(o["object_id"])
            c = (o.get("box_map") or {}).get("center") or o.get("centroid_map")
            if pid and c:
                tracks.setdefault(pid, []).append(c)
    return tracks


def tab_s6_tracking(frames, k):
    owner("S6")
    ids = sorted({p for f in frames for p in f["ids"].values()})
    z, text = [], []
    for pid in ids:
        row, trow = [], []
        for f in frames:
            if pid in f["ids"].values():
                row.append(2)
                trow.append("observed")
            elif any(n["id"] == pid and n["state"] == "occluded" for n in f["graph"]["nodes"]):
                row.append(1)
                trow.append("occluded (kept in memory)")
            else:
                row.append(0)
                trow.append("not present")
        z.append(row)
        text.append(trow)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure(go.Heatmap(z=z, x=[f["frame"] for f in frames], y=ids, text=text, hoverinfo="text+x+y",
                                   colorscale=[[0, "rgba(128,128,128,0.08)"], [0.5, "#e3a44b"], [1, "#3aa463"]],
                                   showscale=False, zmin=0, zmax=2, xgap=2, ygap=2))
        fig.add_vline(x=k, line=dict(color="#2a78d6", width=2))
        fig.update_layout(height=60 + 34 * max(len(ids), 1), margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch", key="s6_timeline")
        st.caption("Persistent-ID timeline: green = observed, orange = occluded but kept in memory. "
                   "Blue line = current frame.")
    with c2:
        st.plotly_chart(topdown(frames[k]["graph"], None, height=360, tracks=tracks_of(frames, k)),
                        width="stretch", key="s6_tracks")
        st.caption("Top-down trajectories up to the current frame.")
    moved = frames[k].get("track_motion_m", {})
    rows = [{"persistent id": p, "moved since last frame (cm)": round(m * 100, 2)} for p, m in sorted(moved.items())]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    events = [{"frame": f["frame"], "event": e["event"], "persistent id": e["persistent_id"]}
              for f in frames[: k + 1] for e in f.get("events", [])]
    if events:
        st.markdown("**Events up to this frame** (APPEARED / MOVED / OCCLUDED / REAPPEARED / PICKED_UP / PLACED_ON / REMOVED)")
        st.dataframe(pd.DataFrame(events[::-1]), hide_index=True, width="stretch", height=220)
    cfg = load_config()["tracking"]
    st.caption(f"Association (`src/tracking/tracker.py`): Hungarian on position (gate {cfg['gate_m'] * 100:.0f} cm after "
               f"velocity prediction) + label + size; hidden objects kept {cfg['max_missed_frames']} frames and "
               f"re-identified up to {cfg.get('reid_gate_m', 0.6) * 100:.0f} cm away. Tracks in this recording: {len(ids)}.")


def tab_s4(frames, k):
    owner("S4")
    g = frames[k]["graph"]
    active = [e for e in g["edges"] if e["end_time"] is None]
    c1, c2 = st.columns([3, 2])
    with c1:
        preds = sorted({e["predicate"] for e in active})
        show = st.multiselect("Relations shown", preds, default=[p for p in preds if p != "left_of"] or preds,
                              key="s4_preds")
        st.plotly_chart(graph_fig(g, show=set(show)), width="stretch", key="s4_graph")
        st.caption("Nodes sit where the objects are on the desk (top-down). Hover a relation label for its "
                   "metric proof. Grey = occluded (kept in memory); square = virtual desk plane.")
    with c2:
        prev = {edge_key(e) for e in frames[k - 1]["graph"]["edges"] if e["end_time"] is None} if k else set()
        now = {edge_key(e) for e in active}
        st.markdown("**Changes since previous frame**")
        if k == 0:
            st.caption("first frame")
        for s, p, o in sorted(now - prev):
            st.markdown(f"<span class='g1-ok'>+ {s} {p} {o}</span>", unsafe_allow_html=True)
        for s, p, o in sorted(prev - now):
            st.markdown(f"<span class='g1-warn'>− {s} {p} {o}</span>", unsafe_allow_html=True)
        if k and now == prev:
            st.caption("no relation changed")
        st.metric("nodes / active edges", f"{len(g['nodes'])} / {len(active)}")
        st.caption(f"frame: {g.get('frame')} · units: {g.get('units')} · desk alignment: "
                   f"{json.dumps(g.get('desk_alignment', 'config'))[:120]}")
    t0 = frames[0].get("timestamp") or 0.0
    rel = lambda t: None if t is None else round(t - t0, 3)
    st.markdown("**Edges with their metric proof** (ended edges are kept as history; times from the first frame)")
    rows = [{"subject": e["source"], "predicate": e["predicate"], "object": e["target"],
             "evidence": ", ".join(f"{a}={b}" for a, b in e["evidence"].items() if a != "frame"),
             "confidence": e.get("confidence"), "start (s)": rel(e.get("start_time")), "end (s)": rel(e.get("end_time")),
             "status": "active" if e["end_time"] is None else "ended"} for e in g["edges"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("Scene Graph JSON (hand-over to S5)"):
        st.json(g, expanded=False)


def demo_queries(graph):
    labels = []
    for n in graph["nodes"]:
        if n.get("virtual"):
            continue
        lab = node_label(n)
        if lab not in labels:
            labels.append(lab)
    qs = [f"Where is the {labels[0]}?"] if labels else []
    if len(labels) > 1:
        qs += [f"What is closest to the {labels[0]}?", f"Go to the {labels[1]}"]
        # negative query: name a real object that is NOT near labels[0], so the verifier has a target to prove
        ref = [n for n in graph["nodes"] if not n.get("virtual") and node_label(n) == labels[0]]
        for lab in labels[1:]:
            cands = [n for n in graph["nodes"] if not n.get("virtual") and node_label(n) == lab]
            if any(all(np.linalg.norm(np.subtract(c["pose_map"], r["pose_map"])) >= 0.3 for r in ref) for c in cands):
                qs.append(f"Find the {lab} that is not near the {labels[0]}")
                break
    qs += ["Where is the portable lidar scanner?"]
    return labels, qs


def ask(question, graph, cfg, parser):
    rr = s5_adapter.answer(question, graph, parser)
    goal = build_action_goal(rr, graph, cfg["action"])
    return {"q": question, "rr": rr, "goal": goal, "sim": simulate(goal, graph, cfg["action"])}


def tab_s5(graph, cfg, parser, hist_key):
    owner("S5")
    st.caption("The LLM only translates the question into strict JSON; Python proves the answer on the S4 graph. "
               "If the geometry cannot prove it, the system says “Cannot determine” instead of guessing.")
    labels, qs = demo_queries(graph)
    cols = st.columns(len(qs))
    preset = None
    for c, q in zip(cols, qs):
        if c.button(q, width="stretch", key=f"q_{q}"):
            preset = q
    question = st.chat_input("Ask about the scene, e.g. 'Which object is closest to the laptop?'") or preset
    if question:
        st.session_state[hist_key].insert(0, ask(question, graph, cfg, parser))
        st.rerun()                      # redraw the other tabs (overview, action) with the new answer
    hist = st.session_state[hist_key]
    if not hist:
        st.info("Pick a question or type one.")
        return
    h = hist[0]
    rr = h["rr"]
    with st.chat_message("user"):
        st.write(h["q"])
    with st.chat_message("assistant"):
        (st.success if rr["verification_status"] else st.warning)(rr["final_answer"])
        for w in rr["warnings"]:
            st.caption(f"⚠ {w}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**1 · Parsed intent** (parser: {rr['parser']})")
        st.json(rr.get("llm_parsed_query", {}), expanded=True)
    with c2:
        st.markdown("**2 · Geometric proof**")
        st.metric("verified", "yes" if rr["verification_status"] else "no", f"confidence {rr['confidence']}")
        st.code(rr["computed_facts"], language="text")
    with st.expander("ReasoningResult JSON (hand-over to S6)"):
        st.json({k: v for k, v in rr.items()}, expanded=False)
    with st.expander("Query suite on this frame"):
        if st.button("Run all suggested questions", key="suite"):
            rows = []
            suite = list(dict.fromkeys(qs + [f"Where is the {lab}?" for lab in labels]))
            for q in suite:
                r = s5_adapter.answer(q, graph, parser)
                rows.append({"question": q, "verified": r["verification_status"], "target": r["target_id"],
                             "confidence": r["confidence"], "answer": r["final_answer"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if len(hist) > 1:
        with st.expander("Earlier questions"):
            for old in hist[1:]:
                st.write(f"**{old['q']}** → {old['rr']['final_answer']}")


def tab_s6_action(run_dir, frame, goal_state):
    owner("S6")
    if not goal_state:
        st.info("Ask a question in the S5 tab first.")
        return
    goal, sim, rr = goal_state["goal"], goal_state["sim"], goal_state["rr"]
    st.markdown(f"**Question:** {goal_state['q']} → target `{rr['target_id']}`")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(topdown(frame["graph"], goal, height=460), width="stretch", key="s6_twin")
        st.caption("Virtual twin (top-down, map frame). Red = exclusion polygons around objects the agent must "
                   "never cross; the path is accepted only if it stays clear (planner: straight, or detour around the zones).")
    with c2:
        st.metric("ActionGoal", "validated ✓" if goal["validated"] else "rejected ✗", goal["reason"],
                  delta_color="normal" if goal["validated"] else "inverse")
        st.json({k: goal.get(k) for k in ("target_object_id", "target_pose", "validated", "reason_code", "reason", "warnings")},
                expanded=True)
        st.markdown("**Virtual-twin simulation**")
        st.json(sim, expanded=True)
    with st.expander("3D view with exclusion zones and path"):
        st.plotly_chart(view_3d(run_dir, frame, goal), width="stretch", key="s6_3d")
    with st.expander("Full ActionGoal JSON"):
        st.json(goal, expanded=False)


def tab_s7(run_dir, summary, frames, ds):
    owner("S7")
    m = st.columns(4)
    m[0].metric("mean latency", f"{summary['mean_latency_ms']:.0f} ms")
    totals = [f["timings_ms"]["total"] for f in frames]
    m[1].metric("p95 latency", f"{np.percentile(totals, 95):.0f} ms")
    m[2].metric("FPS", f"{summary['fps']:.2f}")
    m[3].metric("contract errors", len(summary["contract_errors"]))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Mean latency per stage (ms)**")
        lat = summary["stage_latency_ms"]
        fig = go.Figure(go.Bar(x=list(lat.values()), y=list(lat.keys()), orientation="h", marker_color="#2a78d6"))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width="stretch", key="s7_stage")
    with c2:
        st.markdown("**Per-frame latency (ms)**")
        df = pd.DataFrame([{"frame": f["frame"], **{k: v for k, v in f["timings_ms"].items() if k != "total"}}
                           for f in frames]).set_index("frame")
        st.bar_chart(df, height=260)
    if summary["contract_errors"]:
        st.error("Contract errors:\n" + "\n".join(summary["contract_errors"]))
    else:
        st.success("Every hand-over (S3→S2, S2→S6/S4, S4→S5, S5→S6, S6→twin) passed its JSON contract check.")
    for e in summary.get("frame_errors", []):
        st.warning(f"{e['frame']}: {e['error']}")
    if summary.get("skipped_no_s3"):
        st.warning(f"{len(summary['skipped_no_s3'])} frames had no S3 masks and were skipped.")
    s3 = summary.get("session_meta", {}).get("s3")
    if s3:
        st.caption(f"S3 ↔ bag alignment: {s3.get('alignment')} · {s3.get('matched')} frames with masks")
    c1, c2 = st.columns(2)
    c1.download_button("Download run_summary.json", (run_dir / "run_summary.json").read_bytes(),
                       file_name=f"{ds['name']}_run_summary.json", mime="application/json", width="stretch")
    c2.download_button("Download scene graph (current frame)", json.dumps(frames[-1]["graph"], indent=2),
                       file_name=f"{ds['name']}_scene_graph.json", mime="application/json", width="stretch")
    st.markdown("**Reproduce from the command line**")
    st.code(f"python -m demo.runs --import <package>.zip      # S1 replay + S3 masks -> dataset + pipeline run\n"
            f"python -m demo.runs --rerun {ds['name']}\n"
            f"python main_pipeline.py --frames {runs.DATASETS.relative_to(ROOT) if runs.DATASETS.is_relative_to(ROOT) else runs.DATASETS}/{ds['name']}/frames\n"
            f"python -m pytest -q && python tests/smoke_test.py", language="bash")
    with st.expander("Integration config (configs/s7_integration.yaml)"):
        st.code((ROOT / "configs" / "s7_integration.yaml").read_text(encoding="utf-8"), language="yaml")


# ================================================================ page
def main():
    if not st.session_state.get("_booted"):
        with st.spinner("Preparing the demo recordings…"):
            runs.ensure_builtin()
            try:
                runs.bootstrap_from_env()
            except Exception as e:
                st.sidebar.error(f"DEMO_DATA_URLS import failed: {e}")
        st.session_state._booted = True
    ds, parser, _ = sidebar()
    if ds is None:
        st.warning("No recording yet. Add one in the sidebar.")
        return
    run_dir = ds["run"]
    summary, frames = load_run(str(run_dir), (run_dir / "run_summary.json").stat().st_mtime)
    k, playing, speed = replay_controls(len(frames))
    frame = frames[k]
    cfg = load_config()

    hist_key = f"hist_{ds['name']}_{frame['frame']}"
    if hist_key not in st.session_state:
        st.session_state[hist_key] = []
        lq = summary.get("last_query")
        if lq:
            st.session_state[hist_key].append(ask(lq["question"], frame["graph"], cfg, parser))
    goal_state = st.session_state[hist_key][0] if st.session_state[hist_key] else None

    st.title("Semantic Scene Understanding & Scene-Graph Reasoning")
    st.caption(f"Capstone G1 · ZED Mini RGB-D tabletop micro-world · recording **{ds['name']}** · "
               f"frame **{frame['frame']}** ({k + 1}/{len(frames)})" + (" · ▶ replaying" if playing else ""))
    pages = {
        "Overview": lambda: tab_overview(summary, frames, frame, goal_state),
        "S1 Sensors": lambda: tab_s1(summary, frame),
        "S3 Perception": lambda: tab_s3(frame),
        "S2 3D Fusion": lambda: tab_s2(run_dir, frame, goal_state["goal"] if goal_state else None),
        "S6 Tracking": lambda: tab_s6_tracking(frames, k),
        "S4 Scene Graph": lambda: tab_s4(frames, k),
        "S5 Reasoning": lambda: tab_s5(frame["graph"], cfg, parser, hist_key),
        "S6 Action": lambda: tab_s6_action(run_dir, frame, goal_state),
        "S7 Integration": lambda: tab_s7(run_dir, summary, frames, ds),
    }
    # Only the selected module is drawn, so bag replay stays fast while a page is on the projector.
    page = st.segmented_control("Module", list(pages), default="Overview", key="page",
                                label_visibility="collapsed") or "Overview"
    pages[page]()

    if playing:
        ticker(speed)


def ticker(seconds: float):
    """Advance the replay from a fragment timer: the page never sleeps, so Pause is always honoured."""
    st.session_state._tick = time.time()

    @st.fragment(run_every=seconds)
    def _tick():
        if st.session_state.get("playing") and time.time() - st.session_state.get("_tick", 0) >= 0.9 * seconds:
            st.rerun(scope="app")

    _tick()


main()
