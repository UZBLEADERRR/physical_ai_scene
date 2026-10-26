# G1 cloud demo — every module on one web page

`demo/app.py` is the presentation app. It replays a recorded ZED Mini bag frame by frame (the cloud
stand-in for `ros2 bag play S2_bag`) and shows, for the current frame, what each team member's module
produced:

| Tab | Owner | What you see |
|---|---|---|
| Overview | all | pipeline cards S1 → S7 with this frame's output and latency, camera + top-down twin |
| S1 Sensors | Muzaffar | RGB, depth heat-map, `camera_intrinsics.yaml` (camera_info or fallback), `T_map_camera` (height, pitch), bag topics / rates / sync stats |
| S3 Perception | Li Yizhe | YOLO-World boxes + SAM 2 masks, label candidates, scores, seen / unseen split, mask cut-outs |
| S2 3D Fusion | Sarvar | point cloud + gravity-aligned 3D boxes, centroids, sizes, depth confidence (glass in red), fusion weights, ObjectInstance3D JSON |
| S6 Tracking | Erkinov Shakhzodjon | persistent-ID timeline (observed / occluded), top-down trajectories, motion per frame |
| S4 Scene Graph | Jaloliddin | graph drawn on the desk layout, relation filter, changes since the previous frame, edge table with metric proof and start/end times |
| S5 Reasoning | Kurbonaliyev Sanjarbek | chat, parsed intent JSON, geometric proof, "Cannot determine" refusals, query suite |
| S6 Action | Erkinov Shakhzodjon | ActionGoal (validated / rejected + reason), exclusion zones, virtual-twin simulation |
| S7 Integration | Mirzokhidbek | per-stage and per-frame latency, p95, FPS, contract checks, downloads, reproduce commands |

The sidebar has **▶ Play / ⏸ Pause / ⏮ Start**, a speed selector and a frame slider.

## 1. Get the team's recordings into the demo

The demo needs, per bag: the RGB + depth topics, ideally `camera_info` and `/tf_static`, and S3's
output for that bag (`frame_XXXXXX/observation.json` + `masks/`). It finds the topics itself
(ZED wrapper names, or short names such as `/rgb`, `/depth`, `/tf`). Without `camera_info` it uses
S1's fallback `configs/sensors/camera_intrinsics.yaml`, scaled to the image size. Without TF it uses
`tf_map_camera_link` from `configs/fusion_3d.yaml`. The S1 tab always says which source was used.

**Pair S3 output with its bag by name.** Put the bag name or the scenario tag in the S3 folder name,
e.g. `S2_bag.db3` ↔ `output/s3_semantic/S2_static/...`. The bag is then replayed at exactly the
timestamps S3 looked at: absolute bag time, or video-relative time shifted to the bag's first RGB
frame. If S3 saw a different resolution, its masks and boxes are rescaled.

Bags are large (a 90 s HD720 bag is several GB). Pack them on the laptop that has them:

```bash
pip install -r requirements-demo.txt
python -m demo.pack --src recordings/ --max-frames 40 --out G1_bags.zip
#   recordings/S1_bag.db3  recordings/S2_bag/ (metadata.yaml + .db3) ...
#   recordings/output/s3_semantic/S2_static/frame_000000/observation.json ...
```

The zip holds only the replayed frames (depth as 16-bit mm PNG), typically 50–150 MB for 40 frames.
Then use **one** of these:

* **Upload the pieces separately** in the sidebar (*➕ Add recording*): **1 · S1 bag** = `metadata.yaml` + `.db3` (select both), **2 · S3 output** = S3's zip, **3 · Intrinsics** (optional), then *Import recording*. No packing needed on Railway.
* **Upload** the packed zip: *➕ Add recording → One zip with everything → Import zip*.
* **Link**: put it on Google Drive ("Anyone with the link") and paste the link into *…or a link*.
* **Automatic**: set `DEMO_DATA_URLS=<link1>,<link2>` on the server; it is imported once at start-up.

A small bag can also be uploaded directly as `.db3` or `.zip`. One dataset is created per bag. A bag
without S3 output is reported in a warning and skipped, unless the image was built with YOLO-World
(`WITH_S3_LIVE=1`).

Command-line equivalents (same code):

```bash
python -m demo.runs --import G1_bags.zip       # import + run the pipeline
python -m demo.runs --url https://drive.google.com/file/d/<id>/view
python -m demo.runs                            # list recordings
python -m demo.runs --rerun S2_bag --parser ollama
```

## 2. Run locally

```bash
pip install -r requirements-demo.txt
streamlit run demo/app.py                      # http://localhost:8501
# or with Docker
docker build -t g1-demo . && docker run -p 8080:8080 g1-demo
```

## 3. Deploy on Railway

1. railway.com → *New Project → Deploy from GitHub repo* → `intodakt/physical_ai_scene`, branch with this code.
   Railway reads `railway.json` and builds the `Dockerfile`. The first build takes about 10 minutes.
2. *Settings → Networking → Generate Domain*. Railway sets `PORT` itself.
3. Recommended: *+ New → Volume*, mount path **`/data`**. Imported recordings then survive redeploys.
   Without a volume, uploads are lost on every redeploy and only the built-in recording remains.
4. Optional variables: `DEMO_DATA_URLS` (Drive links, imported at start-up), `OLLAMA_HOST` (a
   reachable Ollama server for the LLM parser; otherwise the offline rule parser is used and the
   verifier is identical).
5. Resources: 2 GB RAM is enough; S2 needs about 0.5 s per frame on CPU.

## 4. Deploy on Google Cloud Run

```bash
gcloud run deploy g1-demo --source . --region asia-northeast3 \
  --memory 4Gi --cpu 2 --timeout 3600 --max-instances 1 --execution-environment gen2 --allow-unauthenticated \
  --set-env-vars DEMO_DATA_URLS="https://drive.google.com/file/d/<id>/view"
```

* Cloud Run limits HTTP/1 request bodies to **32 MB**. Upload only small zips there; for real bags use a
  Drive link or `DEMO_DATA_URLS`.
* The container filesystem is in memory and per instance. Use `--max-instances 1` so every viewer sees
  the same imported recordings. For persistence, mount a Cloud Storage bucket at `/data`
  (`--add-volume name=demo,type=cloud-storage,bucket=<bucket> --add-volume-mount volume=demo,mount-path=/data`).
* Streamlit uses WebSockets; Cloud Run supports them, but keep `--timeout 3600` so sessions are not cut after 5 minutes.

## 5. With or without YOLO-World

The default image replays S3's masks and has no PyTorch. To run S3 live on bags without masks (CPU, slow):

```bash
docker build --build-arg WITH_S3_LIVE=1 -t g1-demo-s3 .
```

## Files

| File | Role |
|---|---|
| `demo/app.py` | Streamlit UI (one tab per module) |
| `demo/runs.py` | store of datasets and runs, built-in recording, Drive/URL import |
| `demo/pack.py` | bags + S3 output → small zip for upload |
| `src/sensors/bag_replay.py` | S1 replay: lone `.db3` or bag folder, topic detection, fallback intrinsics, TF → `T_map_camera` |
| `src/integration/demo_data.py` | finds bags / S3 runs / frame folders in an upload, pairs them, aligns S3 timestamps |
| `Dockerfile`, `railway.json`, `.streamlit/config.toml`, `requirements-demo.txt` | deployment |
| `tests/test_demo_replay.py` | replay, import, pack and pipeline-on-import tests |
