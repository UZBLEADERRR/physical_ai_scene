"""Week 9 evidence for S2 (professor's feedback §5.1–5.3): ablation + confidence calibration.

    python scripts/s2_week9_ablation.py            # run the pipeline, write metrics.json
    python scripts/s2_week9_ablation.py --plot     # figures from metrics.json (needs matplotlib)

Three methods on the same mock frames (seeds × camera profiles × glass failure modes):

    F1  RGB + desk plane, no depth   every object placed from its mask and the desk plane given by the TF
    F2  depth only (Week-5 config)   no flying-pixel filter, no reliability check, no mask+plane fallback
    F2+ adaptive (current config)    reliability check, genuine-pixel cleaning, mask+plane fallback

"Shiny" condition: specular reflection is simulated on the mug by dropping 50 % of its depth
pixels and offsetting another 10 % by 2–8 cm (an assumption, not a measured sensor model).

Calibration: for every F2+ object, the new C_depth = r_valid·exp(−σ_Z/τ) and the old
depth_confidence (valid-pixel ratio) are compared with the box-centre error (Spearman ρ, bins)
and with the risk of trusting raw depth: AUROC for "depth-only error > 2 cm". Fixed vs adaptive
fusion weights change only the fused confidence (geometry is identical), so they are compared the
same way.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "s2" / "week9"

SEEDS = [0, 1, 2, 3, 4]
CASES = [  # key, generate kwargs, shiny mug
    ("rs_holes85", dict(profile="realsense", glass_mode="holes"), False),
    ("zed_see85", dict(profile="zed_mini", glass_mode="see_through"), False),
    ("zed_see100", dict(profile="zed_mini", glass_mode="see_through", glass_failure=1.0), False),
    ("zed_holes100", dict(profile="zed_mini", glass_mode="holes", glass_failure=1.0), False),
    ("rs_shiny", dict(profile="realsense", glass_mode="holes"), True),
    ("zed_shiny", dict(profile="zed_mini", glass_mode="see_through"), True),
]
CATEGORY = {"laptop": "matte", "book": "matte", "multimeter": "small", "mug": "cylinder", "glass": "transparent"}
METHODS = ["F1_rgb_plane", "F2_depth_only", "F2_adaptive"]
RISK_M = 0.02  # "depth would mislead": trusting raw depth (F2 depth only) puts the box > 2 cm off


def _w5_config(load_config):
    c = load_config()
    c["depth"]["flying_pixel_filter"]["enabled"] = False
    c["depth_reliability"]["enabled"] = False
    return c


def desk_plane_from_tf(T):
    """Desk is z = 0 in the map frame → plane (n, d) in the camera frame: R[2]·p + t_z = 0."""
    return np.r_[T[2, :3], T[2, 3]]


def make_shiny(depth_m, mask, rng):
    d = depth_m.copy()
    idx = np.flatnonzero(mask.ravel())
    r = rng.random(len(idx))
    flat = d.ravel()
    flat[idx[r < 0.5]] = np.nan
    spike = idx[(r >= 0.5) & (r < 0.6)]
    flat[spike] = flat[spike] + rng.choice([-1, 1], len(spike)) * rng.uniform(0.02, 0.08, len(spike))
    return d


def f1_estimates(frame, obs, T):
    from src.fusion_3d.mask_plane import estimate_from_mask
    plane = desk_plane_from_tf(T)
    out = {}
    for o in obs:
        est = estimate_from_mask(o.mask, frame.intrinsics, plane, T)
        if est is not None:
            out[o.label] = est["box_map"]["center"]
    return out


def run():
    from src.fusion_3d.pipeline import _resolve_tf, load_config, process_frame, sensor_depth
    from src.preprocessing.frame_loader import load_frame, load_semantic_observations
    from src.preprocessing.mock_data import generate

    cfg_new, cfg_w5 = load_config(), _w5_config(load_config)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for key, kw, shiny in CASES:
            for seed in SEEDS:
                d = generate(Path(tmp) / f"{key}_{seed}", seed=seed, **kw)
                fr = load_frame(d)
                obs = load_semantic_observations(d / "observation.json")
                gt = {g["name"]: np.array(g["center"]) for g in json.loads((d / "ground_truth.json").read_text())["objects"]}
                T = _resolve_tf(fr, cfg_new)
                over = {}
                if shiny:
                    mug = next(o for o in obs if o.label == "mug")
                    for name, c in (("new", cfg_new), ("w5", cfg_w5)):
                        over[name] = make_shiny(sensor_depth(fr, c), mug.mask, rng=np.random.default_rng(1000 + seed))
                est = {"F1_rgb_plane": {k: (v, None) for k, v in f1_estimates(fr, obs, T).items()}}
                for m, c, name in (("F2_depth_only", cfg_w5, "w5"), ("F2_adaptive", cfg_new, "new")):
                    res = process_frame(fr, obs, c, depth_override=over.get(name))
                    est[m] = {o.semantic_label: (o.box_map["center"] if o.box_map else None, o) for o in res.objects}
                fixed = process_frame(fr, obs, cfg_new, fusion_mode="fixed", depth_override=over.get("new"))
                fused_fixed = {o.semantic_label: o.fused_confidence for o in fixed.objects}
                for label in gt:
                    row = {"case": key, "seed": seed, "label": label, "category": "shiny" if shiny and label == "mug" else CATEGORY[label]}
                    for m in METHODS:
                        e = est[m].get(label)
                        row[m] = None if e is None or e[0] is None else round(float(np.linalg.norm(np.array(e[0]) - gt[label])), 5)
                    o = est["F2_adaptive"].get(label, (None, None))[1]
                    if o is not None:
                        row.update(c_depth=o.confidence_terms["c_depth"], sigma_z_m=o.confidence_terms["sigma_z_m"],
                                   r_valid=o.confidence_terms["r_valid"], old_depth_conf=o.depth_confidence,
                                   fused=o.fused_confidence, fused_fixed=fused_fixed.get(label), geometry_source=o.geometry_source)
                    rows.append(row)
                print(key, seed, "done", flush=True)
    return rows


def spearman(a, b):
    ra, rb = (np.argsort(np.argsort(x)).astype(float) for x in (np.asarray(a), np.asarray(b)))
    return float(np.corrcoef(ra, rb)[0, 1])


def auroc(score, positive):
    """P(score of a positive > score of a negative), ties count ½ — no sklearn needed."""
    s, p = np.asarray(score, float), np.asarray(positive, bool)
    pos, neg = s[p], s[~p]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())


def stats(v, n_total):
    v = np.array([x for x in v if x is not None]) * 100
    if not len(v):
        return {"n": 0, "recall": 0.0}
    return {"n": int(len(v)), "recall": round(len(v) / n_total, 3), "mean_cm": round(float(v.mean()), 2),
            "median_cm": round(float(np.median(v)), 2), "p95_cm": round(float(np.percentile(v, 95)), 2),
            "max_cm": round(float(v.max()), 2)}


def summarise(rows):
    cats = ["matte", "small", "cylinder", "shiny", "transparent"]
    table = {}
    for cat in cats + ["all"]:
        sub = [r for r in rows if cat == "all" or r["category"] == cat]
        table[cat] = {m: stats([r[m] for r in sub], len(sub)) for m in METHODS}
    glass = {}
    for key, _, _ in CASES:
        sub = [r for r in rows if r["case"] == key and r["label"] == "glass"]
        glass[key] = {m: stats([r[m] for r in sub], len(sub)) for m in METHODS}
    cal = [r for r in rows if r.get("c_depth") is not None and r["F2_adaptive"] is not None]
    err = [r["F2_adaptive"] for r in cal]
    bins = []
    for lo, hi in ((0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.01)):
        b = [r["F2_adaptive"] * 100 for r in cal if lo <= r["c_depth"] < hi]
        bo = [r["F2_adaptive"] * 100 for r in cal if lo <= r["old_depth_conf"] < hi]
        bins.append({"range": [lo, min(hi, 1.0)],
                     "new_n": len(b), "new_mean_cm": round(float(np.mean(b)), 2) if b else None,
                     "old_n": len(bo), "old_mean_cm": round(float(np.mean(bo)), 2) if bo else None})
    see = [r for r in cal if r["label"] == "glass" and "see" in r["case"]]
    risky = [r for r in cal if r["F2_depth_only"] is not None]
    bad = [r["F2_depth_only"] > RISK_M for r in risky]
    calibration = {
        "n": len(cal),
        "spearman_new_c_depth_vs_error": round(spearman([r["c_depth"] for r in cal], err), 3),
        "spearman_old_depth_conf_vs_error": round(spearman([r["old_depth_conf"] for r in cal], err), 3),
        "spearman_new_c_depth_vs_depth_only_error": round(spearman([r["c_depth"] for r in risky], [r["F2_depth_only"] for r in risky]), 3),
        "spearman_old_depth_conf_vs_depth_only_error": round(spearman([r["old_depth_conf"] for r in risky], [r["F2_depth_only"] for r in risky]), 3),
        "risk_threshold_cm": RISK_M * 100,
        "n_risky": int(sum(bad)),
        "auroc_new_c_depth": round(auroc([1 - r["c_depth"] for r in risky], bad), 3),
        "auroc_old_depth_conf": round(auroc([1 - r["old_depth_conf"] for r in risky], bad), 3),
        "auroc_fused_adaptive": round(auroc([1 - r["fused"] for r in risky], bad), 3),
        "auroc_fused_fixed": round(auroc([1 - r["fused_fixed"] for r in risky], bad), 3),
        "bins": bins,
        "see_through_glass": {"old_depth_conf_mean": round(float(np.mean([r["old_depth_conf"] for r in see])), 3),
                              "new_c_depth_mean": round(float(np.mean([r["c_depth"] for r in see])), 3),
                              "matte_c_depth_mean": round(float(np.mean([r["c_depth"] for r in cal if r["category"] == "matte"])), 3)},
    }
    return {"methods": METHODS, "seeds": SEEDS, "cases": [c[0] for c in CASES], "by_category": table,
            "glass_by_case": glass, "calibration": calibration}


def plot(m, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2, MUTED, GRID = "#52514e", "#8a8984", "#e6e5e0"
    COL = {"F1_rgb_plane": "#eb6834", "F2_depth_only": "#8a8984", "F2_adaptive": "#1baf7a"}
    NAME = {"F1_rgb_plane": "F1 RGB + plane", "F2_depth_only": "F2 depth only (W5)", "F2_adaptive": "F2 adaptive (W9)"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})
    cats = ["matte", "small", "cylinder", "shiny", "transparent"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(cats))
    for i, meth in enumerate(METHODS):
        v = [m["by_category"][c][meth].get("median_cm", np.nan) for c in cats]
        p = [m["by_category"][c][meth].get("p95_cm", np.nan) for c in cats]
        ax.bar(x + (i - 1) * 0.27, v, 0.25, color=COL[meth], label=NAME[meth])
        ax.scatter(x + (i - 1) * 0.27, p, color=COL[meth], marker="_", s=160, zorder=3)
    ax.set_xticks(x, cats)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 8], ["0.5", "1", "2", "5", "8"])
    ax.minorticks_off()
    ax.set_ylabel("box-centre error, cm (bar = median, tick = p95)")
    ax.set_title("Ablation: median and p95 box-centre error by object type")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.savefig(OUT / "ablation_by_category.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    cal = [r for r in rows if r.get("c_depth") is not None and r["F2_depth_only"] is not None]
    c = m["calibration"]
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for ax, k, title in ((axs[0], "old_depth_conf", f"Old: valid-pixel ratio  (AUROC {c['auroc_old_depth_conf']:.2f})"),
                         (axs[1], "c_depth", f"New: C_depth  (AUROC {c['auroc_new_c_depth']:.2f})")):
        for cat, col in (("matte", "#2a78d6"), ("small", "#8b7fe0"), ("cylinder", "#52514e"), ("shiny", "#e3a33b"), ("transparent", "#e34948")):
            s = [r for r in cal if r["category"] == cat]
            ax.scatter([r[k] for r in s], [r["F2_depth_only"] * 100 for r in s], s=16, color=col, label=cat, alpha=0.8)
        ax.axhline(RISK_M * 100, color="#8a8984", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("confidence")
        ax.set_xlim(-0.03, 1.03)
        ax.grid(color=GRID, lw=0.6)
    axs[0].set_ylabel("error if raw depth is trusted, cm")
    axs[1].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "calibration_scatter.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("figures written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.plot:
        plot(json.loads((OUT / "metrics.json").read_text()), json.loads((OUT / "rows.json").read_text()))
    else:
        rows = run()
        (OUT / "rows.json").write_text(json.dumps(rows, indent=1))
        m = summarise(rows)
        (OUT / "metrics.json").write_text(json.dumps(m, indent=2))
        print(json.dumps(m["by_category"]["all"], indent=1))
        print(json.dumps(m["calibration"], indent=1))
