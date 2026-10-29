# S6 — tracking evaluation (Week-5 baseline vs MultiTermTracker)

Reproduce: `python scripts/s6_tracking_eval.py`. Means over 10 noise seeds, 30 frames per scenario.
The baseline emits no events, so its event columns are 0 by construction.

| Scenario | Tracker | ID switches | re-ID ok / gaps | events found | event delay (frames) |
|---|---|---|---|---|---|
| static | baseline | 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | — |
| static | multi_term | 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | — |
| relocation | baseline | 1.0 | 0.0 / 1.0 | 0.0 / 3.0 | — |
| relocation | multi_term | 0.0 | 1.0 / 1.0 | 3.0 / 3.0 | 0.0 |
| occlusion | baseline | 0.0 | 1.0 / 1.0 | 0.0 / 2.0 | — |
| occlusion | multi_term | 0.0 | 1.0 / 1.0 | 2.0 / 2.0 | 0.0 |
| slide | baseline | 0.0 | 0.0 / 0.0 | 0.0 / 1.0 | — |
| slide | multi_term | 0.0 | 0.0 / 0.0 | 0.9 / 1.0 | 1.0 |
| crossing | baseline | 0.4 | 0.0 / 0.0 | 0.0 / 0.0 | — |
| crossing | multi_term | 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | — |

Real S1 static bag (3 frames of real S2 output): baseline same IDs on every frame = True (8 IDs); multi-term = True (8 IDs).

Limits: the scenarios are synthetic object lists (no rendering, no S3/S2 errors); real relocation and
occlusion bags still have to be labelled with ground-truth identities.
