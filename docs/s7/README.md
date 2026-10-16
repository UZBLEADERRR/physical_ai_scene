# S7 — Integration, Launch & Dashboard

**Owner:** Student 7 (Mirzokhidbek). S7 builds no AI feature: it makes everyone's code run as one
system, checks every hand-over, measures the system, and shows it on one screen.

## One command per job

| Job | Command |
|---|---|
| Install | `pip install -r requirements.txt` (or `conda env create -f environment.yml`) |
| Whole pipeline (mock desk sequence) | `python main_pipeline.py` |
| Whole pipeline on S1 frames / bag | `python main_pipeline.py --frames data/frames/<seq>` · `python main_pipeline.py --bag data/bags/<bag> --every 10 --s3 live` |
| Ask a different question | `python main_pipeline.py --query "Go to the glass"` |
| One-click smoke test (bag in → ActionGoal out) | `python tests/smoke_test.py` |
| Smoke test on S1's 90 s bag | `python tests/smoke_test.py --bag data/bags/<bag> --s3 live --every 10 --camera-tf <frame_meta.json>` |
| Experiment matrix (metrics spreadsheet) | `python -m src.evaluation.metrics` → `outputs/evaluation/experiment_matrix.xlsx` |
| Dashboard | `streamlit run dashboard/app.py` |
| PR gatekeeper | `python tools/check_contracts.py` (all tracked files) or `python tools/check_contracts.py src/<module>` |
| All tests | `python -m pytest -q` and `cd src/reasoning && python test_verifier.py` |

All paths are relative to the repository root. CI (`.github/workflows/ci.yml`) runs the gatekeeper,
the tests, the smoke test and the experiment matrix on every push and pull request.

## Pipeline order (`main_pipeline.py` → `src/integration/pipeline.py`)

```
S1 frames/bag ─► S3 masks ─► S2 ObjectInstance3D ─► S6 persistent IDs ─► S4 scene graph
                                                                              │
              virtual twin ◄─ S6 ActionGoal ◄─ S5 ReasoningResult ◄─ question ─┘
```

| Stage | Code that runs (owner's own module) | Output in `outputs/pipeline/` |
|---|---|---|
| S1 | `src/preprocessing/bag_reader.py` (bag → frame folders), `frame_loader.load_frame` | `frames/` (bag mode) |
| S3 | `replay`: the `observation.json` in each frame folder · `live`: `src/open_vocab` PerceptionPipeline | `s3/` (live mode) |
| S2 | `src/fusion_3d/pipeline.process_frame` | `objects/<frame>_objects.json`, `points/<frame>.npz` |
| S6 tracking | `src/integration/s6_placeholder.CentroidTracker` (until S6 merges) | `persistent_id` in the objects file |
| S4 | `src/scene_graph/engine.SceneGraphEngine` | `scene_graph/<frame>_scene_graph.json`, `latest.json` |
| S5 | `src/reasoning/main.answer_question` via `src/integration/s5_adapter.py` | `reasoning/last_query.json` |
| S6 action | `src/integration/s6_placeholder.build_action_goal` + `simulate` | in `last_query.json` |
| S7 | timings, contract checks | `run_summary.json` |

S7 never edits a teammate's module: `s5_adapter.py` hands S5's `answer_question` the live S4 graph
instead of a file, and swaps in an offline rule parser (`src/integration/rule_parser.py`, same JSON
intent, run through S5's own `_normalize`) when no Ollama server answers. The dashboard and every
report say which parser was used (`s5_parser`).

## Mock data until S1's bags arrive

`scripts/make_demo_sequence.py` uses S2's ray-cast generator to write an 8-frame desk sequence:
the mug slides 1.3 cm per frame towards the laptop, the book is taken away in frames 4–5 and put back.
That exercises tracking (`mug_01` keeps its ID, `book_01` comes back), occlusion (S4 marks the book
`occluded`) and time (the `mug near laptop` edge appears in frame 3, `book near mug` ends in frame 7).
The smoke test writes these frames into a real ROS 2 bag with S1's ZED Mini topics and reads it back,
so the bag path is tested end to end.

## Dashboard (`dashboard/app.py`)

1. **Camera** — RGB with S3 masks and boxes, labelled with S6's persistent IDs and S2's depth warnings (glass: `transparent_suspected`).
2. **3D view** — S1 depth as a coloured point cloud (map frame), S2's gravity-aligned boxes, S6 exclusion zones and the agent path.
3. **Scene graph** — pyvis network of S4's graph; hover an edge for its metric proof; occluded nodes are grey.
4. **Chat** — the question, then side by side: S5's parsed JSON · S5's geometric proof · S6's ActionGoal, plus a top-down map of the virtual twin (green = accepted path, red dashed = rejected).

A frame slider replays the sequence; the sidebar shows stage latency and which IDs moved.

## Metrics (Section 24) — current numbers, mock sequence, offline parser

`python -m src.evaluation.metrics` with `configs/s7_eval_queries.json` (10 questions, 6 with a wrong LLM answer):

| Metric | Value | How it is computed |
|---|---|---|
| Pipeline latency S1→S6 | ≈ 840 ms/frame (≈ 1.2 FPS, CPU) | mean of per-frame stage timings; S2 fusion is ≈ 97 % of it |
| Sim-to-real success | 5 / 5 | validated goals the virtual agent reached (≤ 5 cm) without entering an exclusion zone |
| Unsafe goals rejected | 2 / 2 | "go to the glass" and "closest to the laptop" cross the laptop zone → `validated: false` |
| Hallucination catch rate | 6 / 6 | wrong LLM answers (`llm_claim`) that the verifier overrode or refused |
| Answer accuracy | 10 / 10 | verifier target label = expected label (or correctly "Cannot determine") |
| Tracking ID switches | 0 | objects that received more than one persistent ID |
| Contract errors | 0 | hand-overs that broke a JSON contract |

These are integration numbers on synthetic data with a small hand-written question set, so they
show that the pipeline is wired correctly, not how accurate it is on the real desk. The real numbers
come from the same command on S1's recordings with `--parser ollama` and `--s3 live` (Weeks 9–14).

## JSON contracts checked at every hand-over (`src/integration/contracts.py`)

| Hand-over | Required fields |
|---|---|
| S3 → S2 observation | `schema_version`, `frame_id`, `image_size`, `detections[]` (`detection_id`, `bbox` xyxy, `labels`, `detector_score`, `mask_status`, relative `mask`) |
| S2 → S4/S6 objects | `objects[]` (`centroid`, `box_dimensions`, `semantic_label`, `depth_confidence` ∈ [0, 1], `object_id`, finite numbers) |
| S4 → S5 scene graph | `frame`, `units`, `nodes[]` (`id`, `names`, `pose_map`, `size`, `state`, `confidence`), `edges[]` (`source`/`target` must be nodes, `predicate`, `evidence`, `start_time`, `end_time`) |
| S5 → S6 reasoning result | `target_id`, `computed_facts`, `verification_status`, `final_answer`, `confidence`, `warnings`; verified ⇔ `target_id` set |
| S6 → twin ActionGoal | `target_object_id`, `target_pose` `[x, y, yaw]`, `exclusion_zones[]` (`polygon` ≥ 3 points), `validated` |

## Pull-request review checklist (gatekeeper)

1. Branch is `sN-<module>` and only touches the owner's folders (+ `requirements.txt`, tests, docs).
2. `python tools/check_contracts.py` → `OK` (no absolute paths, JSON files honour their contract).
3. `python -m pytest -q` and `python tests/smoke_test.py` pass on the branch merged with `main`.
4. New packages are added to `requirements.txt`; no large data (bags, weights) committed — they go to the shared Drive.
5. The module README says how to run it and what it reads / writes.
6. If anything fails: request changes on the PR with the failing lines; merge only when green.

## Integration findings for the team (from merging all branches, Week 6)

| Module | Finding | Suggested fix |
|---|---|---|
| S1 | No `s1-sensors` branch yet. The pipeline needs `camera_intrinsics.yaml` and `T_map_camera` (or the static TF in `configs/fusion_3d.yaml`) for real bags. | Push the recording scripts and one short bag + calibration to Drive. |
| S3 | Stray empty file `tests/test data` (space in the name). S3 has no README with run commands. | Delete the file; add `src/open_vocab/README.md`. |
| S2 | README still points at `src/open_vocab/open_vocab/contract.py`; S3 moved it to `src/open_vocab/contract.py`. | Update the path. |
| S4 | Nodes are exported without `yaw`, but `size` keeps S2's box axes (not map x/y). Downstream footprints are therefore approximate; S7's exclusion zones use the longer side to stay safe. | Export `yaw` per node (S5 ignores unknown fields). |
| S5 | The virtual `desk_01` node competes in "closest object" questions (e.g. "what is closest to the book?" can answer `desk_01`). | Skip `virtual` nodes as candidates unless the question names the desk. |
| S5 | Flat imports (`from verifier import …`) prevent `import src.reasoning`. S7 loads it through `s5_adapter.py`. | Optional: switch to relative imports. |
| S6 | No branch yet. `src/integration/s6_placeholder.py` implements the Week-5 behaviour (15 cm gate, N-frame memory, straight path, 4-field ActionGoal) so the system runs today. | Merge `src/tracking` + `src/planning` with the same inputs/outputs; S7 will switch the launch file to them. |
