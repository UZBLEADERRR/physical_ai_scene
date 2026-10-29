# S3 — alias table and seen / unseen evaluation (reference tools, to be reviewed by S3)

> **Authorship.** The S3 perception pipeline (`src/open_vocab/pipeline.py`, `models.py`, `run.py`,
> `contract.py`, prompt sets) is Li Yizhe's. The files below were added with Claude Code (AI) as a
> reference for roadmap §11.3 P1 and the W6–7 feedback §6; they do not change the pipeline or its
> output. S3 reviews and edits them, labels the frames and reports the numbers.

| File | What |
|---|---|
| `src/open_vocab/configs/aliases.yaml` | canonical name → aliases (mug / cup, multimeter / digital multimeter, …) |
| `src/open_vocab/configs/prompt_variants.yaml` | 5 prompt types per held-out class: exact, synonym, alternative, functional, category |
| `src/open_vocab/aliases.py` | `canonical()`, `canonical_candidates()`, `ambiguity()` (top-1 / top-2 margin) |
| `src/open_vocab/evaluate.py` | P / R / F1 per class and per split (seen / unseen), IoU ≥ 0.5 + canonical label; prompt-robustness table; `--draft-gt` |

Workflow for the W12 table:

```bash
# 1. S3 output on the unseen-object bag (as before)
python -m src.open_vocab.run --images <S4 bag frames> --config src/open_vocab/configs/generic_tabletop.json ...
# 2. draft labels from that output, then correct every file by hand and set "verified": true
python -m src.open_vocab.evaluate --draft-gt --pred output/s3_semantic/S4_bag/ --out labels/S4_bag/
# 3. the table (warns while any label file is unverified)
python -m src.open_vocab.evaluate --gt labels/S4_bag/ --pred output/s3_semantic/S4_bag/ --out reports/s3/
# 4. prompt robustness: one run per prompt type (prompt_variants.yaml), same labels
python -m src.open_vocab.evaluate --gt labels/S4_bag/ --pred exact=out/exact synonym=out/synonym \
    alternative=out/alternative functional=out/functional category=out/category --out reports/s3/
```

The drafted labels inherit S3's own mistakes until a person corrects them; only verified labels give
a real precision / recall. Tests: `tests/test_s3_aliases_eval.py`.
