"""S3 alias normalisation and label ambiguity (roadmap §11.3 P1).

    canonical("coffee cup") -> "mug"           (configs/aliases.yaml)
    ambiguity(detection)    -> top-1 / top-2 margin of label_scores (small margin = ambiguous label)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ALIASES = Path(__file__).resolve().parent / "configs" / "aliases.yaml"
PROMPT_VARIANTS = Path(__file__).resolve().parent / "configs" / "prompt_variants.yaml"


@lru_cache(maxsize=4)
def alias_table(path: str = str(ALIASES)) -> dict:
    """alias (lower case) -> canonical name."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out = {}
    for canon, names in raw.items():
        for n in [canon, *(names or [])]:
            key = n.strip().lower()
            if key in out and out[key] != canon:
                raise ValueError(f"alias {n!r} maps to both {out[key]!r} and {canon!r}")
            out[key] = canon
    return out


def canonical(label: str, path: str = str(ALIASES)) -> str:
    key = label.strip().lower()
    return alias_table(path).get(key, key)


def canonical_candidates(labels: list, path: str = str(ALIASES)) -> list:
    """Label candidates with aliases merged, order kept (["cup", "coffee mug"] -> ["mug"])."""
    out = []
    for lab in labels:
        c = canonical(lab, path)
        if c not in out:
            out.append(c)
    return out


def ambiguity(detection: dict, margin: float = 0.1) -> dict:
    """Top-1 vs top-2 label: ambiguous when two *different* canonical labels score within ``margin``."""
    labels, scores = detection.get("labels") or [], detection.get("label_scores") or []
    pairs = []
    for lab, s in zip(labels, scores):
        c = canonical(lab)
        if all(c != p[0] for p in pairs):
            pairs.append((c, float(s)))
    if len(pairs) < 2:
        return {"top1": pairs[0][0] if pairs else (canonical(labels[0]) if labels else None), "top2": None,
                "margin": None, "ambiguous": False}
    (l1, s1), (l2, s2) = pairs[0], pairs[1]
    return {"top1": l1, "top2": l2, "margin": round(s1 - s2, 4), "ambiguous": (s1 - s2) < margin}


def prompt_variants(path: str = str(PROMPT_VARIANTS)) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
