"""S3 seen / unseen evaluation: precision, recall, F1 per class and per split (roadmap §11.3, W6–7 §6).

Ground truth: one JSON per frame (hand-labelled; ``--draft-gt`` writes a first version from S3's
own output that a person then corrects and marks ``"verified": true``)::

    {"frame_id": "frame_000130", "verified": true,
     "objects": [{"label": "digital multimeter", "bbox": [x1, y1, x2, y2], "split": "unseen"}, ...]}

Predictions: S3's observation.json files. A prediction matches a ground-truth box when the
canonical labels agree (configs/aliases.yaml) and IoU >= 0.5; each box matches at most once,
highest score first. Detections that match nothing are false positives (a calculator called a
multimeter is a distractor false positive).

    python -m src.open_vocab.evaluate --gt labels/S4_bag/ --pred output/s3_semantic/S4_bag/ --out reports/s3/
    python -m src.open_vocab.evaluate --draft-gt --pred output/s3_semantic/S4_bag/ --out labels/S4_bag/
    # prompt robustness: one S3 run per prompt type, compared on the same labels
    python -m src.open_vocab.evaluate --gt labels/S4_bag/ --pred exact=out/exact synonym=out/synonym ... --out reports/s3/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .aliases import canonical

UNSEEN_FILE = Path(__file__).resolve().parent / "configs" / "unseen_objects.txt"


def unseen_classes(path=UNSEEN_FILE) -> set:
    return {canonical(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_frame(gt_objects: list, detections: list, iou_min: float = 0.5, min_score: float = 0.0) -> dict:
    """Per canonical class: tp / fp / fn for one frame."""
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    gts = [(canonical(g["label"]), g["bbox"]) for g in gt_objects]
    used = [False] * len(gts)
    dets = sorted((d for d in detections if d.get("detector_score", 1.0) >= min_score),
                  key=lambda d: -d.get("detector_score", 1.0))
    for d in dets:
        lab = canonical(d["labels"][0])
        best, best_iou = None, iou_min
        for k, (gl, gb) in enumerate(gts):
            if not used[k] and gl == lab:
                v = iou(d["bbox"], gb)
                if v >= best_iou:
                    best, best_iou = k, v
        if best is None:
            counts[lab]["fp"] += 1
        else:
            used[best] = True
            counts[lab]["tp"] += 1
    for k, (gl, _) in enumerate(gts):
        if not used[k]:
            counts[gl]["fn"] += 1
    return counts


def prf(c) -> dict:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {**c, "precision": None if p is None else round(p, 4), "recall": None if r is None else round(r, 4),
            "f1": None if f is None else round(f, 4)}


def evaluate(gt: dict, pred: dict, unseen=None, iou_min=0.5, min_score=0.0) -> dict:
    """``gt``: frame_id -> list of objects; ``pred``: frame_id -> list of detections."""
    unseen = unseen if unseen is not None else unseen_classes()
    total = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for fid, objs in gt.items():
        for lab, c in match_frame(objs, pred.get(fid, []), iou_min, min_score).items():
            for k in c:
                total[lab][k] += c[k]
    per_class = {lab: {**prf(c), "split": "unseen" if lab in unseen else "seen"} for lab, c in sorted(total.items())}
    splits = {}
    for split in ("seen", "unseen"):
        agg = {"tp": 0, "fp": 0, "fn": 0}
        for c in per_class.values():
            if c["split"] == split:
                for k in agg:
                    agg[k] += c[k]
        splits[split] = prf(agg)
    return {"frames": len(gt), "iou_min": iou_min, "min_score": min_score, "per_class": per_class, "splits": splits}


# ---------------------------------------------------------------- files
def load_gt(folder) -> dict:
    out = {}
    for p in sorted(Path(folder).glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[d["frame_id"]] = d["objects"]
    return out


def load_pred(folder) -> dict:
    out = {}
    for p in sorted(Path(folder).rglob("observation.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[d["frame_id"]] = d["detections"]
    return out


def draft_gt(pred_folder, out_folder, min_score=0.25) -> int:
    """First draft of the labels from S3's own output — every file needs a human check."""
    unseen, out = unseen_classes(), Path(out_folder)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for fid, dets in load_pred(pred_folder).items():
        objs = [{"label": canonical(d["labels"][0]), "bbox": d["bbox"],
                 "split": "unseen" if canonical(d["labels"][0]) in unseen else "seen"}
                for d in dets if d.get("detector_score", 1) >= min_score]
        (out / f"{fid}.json").write_text(json.dumps({"frame_id": fid, "verified": False, "objects": objs}, indent=2))
        n += 1
    return n


def report_md(res: dict, robustness: dict | None = None) -> str:
    L = ["| Split | TP | FP | FN | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|"]
    for s, c in res["splits"].items():
        L.append(f"| {s} | {c['tp']} | {c['fp']} | {c['fn']} | {c['precision']} | {c['recall']} | {c['f1']} |")
    L += ["", "| Class | Split | TP | FP | FN | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|---|"]
    for lab, c in res["per_class"].items():
        L.append(f"| {lab} | {c['split']} | {c['tp']} | {c['fp']} | {c['fn']} | {c['precision']} | {c['recall']} | {c['f1']} |")
    if robustness:
        L += ["", "| Prompt type | Unseen recall | Unseen precision | Unseen F1 |", "|---|---|---|---|"]
        for k, r in robustness.items():
            u = r["splits"]["unseen"]
            L.append(f"| {k} | {u['recall']} | {u['precision']} | {u['f1']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", help="folder of ground-truth JSON files")
    ap.add_argument("--pred", nargs="+", required=True, help="S3 output folder, or several name=folder (prompt types)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--draft-gt", action="store_true", help="write draft labels from --pred into --out")
    a = ap.parse_args()
    if a.draft_gt:
        print(f"{draft_gt(a.pred[0], a.out)} draft label files in {a.out} — check and correct each one, then set verified: true")
        return
    gt = load_gt(a.gt)
    unverified = [f for f in Path(a.gt).glob("*.json") if not json.loads(f.read_text()).get("verified")]
    if unverified:
        print(f"WARNING: {len(unverified)} label files are not verified by a person yet")
    runs = dict(p.split("=", 1) if "=" in p else ("default", p) for p in a.pred)
    results = {k: evaluate(gt, load_pred(v), iou_min=a.iou, min_score=a.min_score) for k, v in runs.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "s3_eval.json").write_text(json.dumps(results, indent=2))
    first = next(iter(results.values()))
    md = report_md(first, results if len(results) > 1 else None)
    (out / "s3_eval.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
