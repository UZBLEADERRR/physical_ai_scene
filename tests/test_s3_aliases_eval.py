"""S3: alias normalisation, label ambiguity and seen / unseen P / R / F1."""

import json

import pytest

from src.open_vocab.aliases import ambiguity, canonical, canonical_candidates, prompt_variants
from src.open_vocab.evaluate import draft_gt, evaluate, iou, load_gt, load_pred, match_frame, unseen_classes


def test_aliases():
    assert canonical("Coffee Cup ") == canonical("cup") == "mug"
    assert canonical("multimeter") == canonical("voltage meter") == "digital multimeter"
    assert canonical("stapler") == "stapler"                     # unknown names pass through
    assert canonical_candidates(["cup", "coffee mug", "mug"]) == ["mug"]


def test_every_unseen_class_has_five_prompt_types():
    v = prompt_variants()
    assert set(v) == unseen_classes()
    assert all(set(p) == {"exact", "synonym", "alternative", "functional", "category"} for p in v.values())
    assert all(canonical(t) == cls for cls, p in v.items() for t in p.values())


def test_ambiguity():
    assert not ambiguity({"labels": ["laptop", "notebook computer"], "label_scores": [0.9, 0.5]})["ambiguous"]
    a = ambiguity({"labels": ["mug", "drinking glass"], "label_scores": [0.41, 0.38]})
    assert a["ambiguous"] and a["top2"] == "drinking glass" and a["margin"] == pytest.approx(0.03)


def _det(label, box, score=0.8):
    return {"labels": [label], "bbox": box, "detector_score": score}


def test_matching_counts_tp_fp_fn():
    gt = [{"label": "digital multimeter", "bbox": [10, 10, 50, 50]}, {"label": "mug", "bbox": [100, 10, 140, 60]}]
    dets = [_det("multimeter", [12, 11, 51, 49]),                # alias + IoU > 0.5 -> TP
            _det("digital multimeter", [200, 10, 240, 50]),      # distractor (calculator) -> FP
            _det("cup", [10, 10, 50, 50])]                        # right label, wrong place -> FP; mug -> FN
    c = match_frame(gt, dets)
    assert c["digital multimeter"] == {"tp": 1, "fp": 1, "fn": 0}
    assert c["mug"] == {"tp": 0, "fp": 1, "fn": 1}
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_seen_unseen_split():
    gt = {"f0": [{"label": "digital multimeter", "bbox": [10, 10, 50, 50]}, {"label": "mug", "bbox": [100, 10, 140, 60]}],
          "f1": [{"label": "soldering iron", "bbox": [10, 10, 90, 30]}]}
    pred = {"f0": [_det("multimeter", [10, 10, 50, 50]), _det("mug", [100, 10, 140, 60])], "f1": []}
    r = evaluate(gt, pred)
    assert r["splits"]["seen"]["recall"] == 1.0
    assert r["splits"]["unseen"]["recall"] == 0.5 and r["splits"]["unseen"]["precision"] == 1.0
    assert r["per_class"]["soldering iron"]["split"] == "unseen"


def test_draft_gt_round_trip(tmp_path):
    run = tmp_path / "s3" / "frame_000000"
    run.mkdir(parents=True)
    obs = json.loads(open("data/mock/frame_000/observation.json", encoding="utf-8").read())
    (run / "observation.json").write_text(json.dumps(obs))
    assert draft_gt(tmp_path / "s3", tmp_path / "labels") == 1
    gt = load_gt(tmp_path / "labels")
    r = evaluate(gt, load_pred(tmp_path / "s3"))
    assert r["splits"]["seen"]["precision"] == 1.0                      # S3 against its own draft
