# Real-Time Re-Identification in Distributed Camera Networks

Implementation of the system described in the thesis of the same name. An edge–server architecture that runs YOLO detection and OSNet embedding on each camera feed, matches identities across cameras via cosine similarity, and persists every sighting to PostgreSQL.

---

## Architecture

```
Camera feeds
    │
    ▼
[Edge client]  ──  YOLO v8n detection
(edge_client.py)   OSNet-AIN-x1.0 embedding
    │               POST /register_track
    ▼
[Central server]  ──  cosine-similarity gallery match
(server.py)           EMA feature-vector update
    │                 PostgreSQL persistence
    ▼
[Dashboard]       ──  Streamlit live view
(dashboard.py)        identity grid + sightings table
```

The edge client runs the full detection–embedding pipeline locally; only a single float32 feature vector (512 dims) and a JPEG crop are sent to the server per tracked object. This keeps network traffic constant regardless of image resolution.

---

## Key results

All numbers are from the evaluation in the thesis (Section 5 / Appendix A). Reproduced by running `evaluate_system.py` against the live system.

| Metric | Thesis result | CI gate |
|---|---|---|
| Server mean latency | **15.6 ms** | ≤ 20 ms |
| Server throughput (sequential) | **64.3 req/s** | ≥ 55 req/s |
| Edge inference (GPU) | **102.2 FPS** | ≥ 90 FPS |
| Detection budget (CPU, YOLOv8n) | ~30 ms/frame | ≤ 40 ms |

The nightly CI job (`integration.yml`) fails the build if any gate regresses, so these are live numbers, not a one-time measurement.

---

## Repository layout

```
server.py               Central FastAPI server (port 5005)
edge_client.py          Edge detection + embedding client
dashboard.py            Streamlit monitoring dashboard (port 8501)
start.py                One-command launcher for the full stack
evaluate_system.py      12-phase integration test suite
sim_cameras.py          Generate synthetic camera feeds from dataset images
demo_system.py          Composite demo video (3-camera + dashboard overlay)
requirements.txt        Python dependencies
docker/
  Dockerfile.server     Slim server image (no torch)
  Dockerfile.edge       Edge image with CPU torch + torchreid + ultralytics
.github/workflows/
  ci.yml                Lint + unit tests + Docker build on every push
  integration.yml       Full 12-phase suite nightly on self-hosted GPU runner
```

---

## Setup

### Prerequisites

- Python 3.12 (the system runs on 3.14 locally; 3.12 is the CI baseline)
- PostgreSQL 16 running locally
- CUDA-capable GPU recommended for the edge client (falls back to CPU)

### 1. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # or /cpu
pip install -r requirements.txt
```

### 2. PostgreSQL

The server connects with these hardcoded credentials (see `server.py:DB_CONFIG`):

```
host:     127.0.0.1
port:     5432
database: reid_logs
user:     postgres
password: 707070
```

Create the database once:

```sql
CREATE DATABASE reid_logs;
```

Tables (`identities`, `sightings`) are created automatically on first server start.

### 3. Datasets (required for sim_cameras.py and demo_system.py only)

Place datasets under `DATA/`:

```
DATA/
  AIC21_Track2_ReID/
    image_test/          # vehicle crops
    image_train/
    train_label.xml
  Market-1501/
    Market-1501-v15.09.15/
      bounding_box_test/ # person crops
      bounding_box_train/
```

The datasets are not tracked in this repository. Download links are in the thesis (Section 3.2).

### 4. Model weights

Place the fine-tuned vehicle checkpoint in the repo root:

```
vehicle_reid_osnet_v4.pth
```

The person model (`osnet_ain_x1_0`, Market-1501 pretrained) is downloaded automatically by torchreid on first run.

---

## Running

### Quick start — full stack

```bash
python start.py          # server + Streamlit dashboard, opens browser
python start.py --demo   # + 3 edge clients on sim_cam1/2/3.mp4 (generate first)
```

### Step by step

```bash
# 1. Generate simulation videos (needs datasets in DATA/)
python sim_cameras.py

# 2. Start the server
python server.py

# 3. Start the dashboard (separate terminal)
streamlit run dashboard.py

# 4. Start an edge client (separate terminal)
python edge_client.py --camera-id Camera-1 --video-source sim_cam1.mp4

# 5. Or run the full demo video
python demo_system.py
```

### Options

**`edge_client.py`**

| Flag | Default | Description |
|---|---|---|
| `--camera-id` | `Camera-1` | Identifier sent to the server |
| `--video-source` | `sim_cam1.mp4` | Video file or webcam index |
| `--server-url` | `http://127.0.0.1:5005/register_track` | Server endpoint |
| `--weight-path` | `vehicle_reid_osnet_v4.pth` | Vehicle OSNet checkpoint |
| `--save-crops` | off | Write best crop per track to `--crops-dir` |

**`start.py`**

| Flag | Description |
|---|---|
| `--demo` | Launch 3 edge clients on sim_cam1/2/3.mp4 |
| `--video FILE` | Launch one edge client on a custom video |
| `--no-browser` | Do not open the dashboard automatically |

---

## Evaluation

Run the full 12-phase suite against a live server:

```bash
python server.py &         # server must be running first
python evaluate_system.py
python evaluate_system.py --no-pipeline   # skip video processing (phases 7–8)
python evaluate_system.py --report out.txt
```

**Phases**

| # | What is tested |
|---|---|
| 1 | Server health & FastAPI reachability |
| 2 | PostgreSQL persistence (`identities` + `sightings` tables) |
| 3 | All REST API endpoints |
| 4 | Match-or-create logic & EMA gallery update |
| 5 | Threshold config hot-reload (`PATCH /config`) |
| 6 | ReID model pipeline (models load, embeddings are valid) |
| 7 | Full pipeline via sim_cam videos (YOLO → embedding → POST) |
| 8 | Cross-camera re-identification |
| 9 | Gallery persistence (feature vectors stored and reloaded from DB) |
| 10 | Dashboard reachability |
| 11 | POST `/register_track` latency & throughput |
| 12 | Real-time pipeline throughput (edge FPS benchmark) |

---

## API reference

The server exposes a FastAPI application documented at `http://localhost:5005/docs`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/register_track` | Submit a detection; returns `global_id`, `score`, `is_new` |
| `GET` | `/stats` | Gallery size, identity counts, active cameras |
| `GET` | `/identities` | List all known identities |
| `GET` | `/identities/{id}/crop` | Best stored crop image (JPEG) |
| `GET` | `/identities/{id}/sightings` | Sighting history for one identity |
| `PATCH` | `/config` | Update matching thresholds at runtime |
| `POST` | `/gallery/reset` | Clear gallery and all database records |
| `POST` | `/camera/{id}/frame` | Push a live frame for dashboard view |
| `GET` | `/camera/{id}/frame` | Retrieve latest frame for a camera |
| `GET` | `/cameras` | List cameras that have pushed frames |
| `GET` | `/get_log` | Download full sightings log as CSV |

**`POST /register_track` payload**

```json
{
  "camera_id":  "Camera-1",
  "yolo_class": 2,
  "features":   [0.012, -0.034, ...],
  "crop_b64":   "<base64-encoded JPEG>"
}
```

`yolo_class`: `0` = person, `1` = bicycle, `2` = car, `3` = motorcycle.

---

## Docker

```bash
# Server (no GPU needed)
docker build -f docker/Dockerfile.server -t reid-server .
docker run -p 5005:5005 reid-server

# Edge client (CPU inference)
docker build -f docker/Dockerfile.edge -t reid-edge .
docker run reid-edge python edge_client.py \
  --video-source /data/feed.mp4 \
  --server-url http://host.docker.internal:5005/register_track

# Edge client with GPU passthrough
docker run --gpus all reid-edge python edge_client.py ...
```

---

## CI

Every push runs three jobs (see `.github/workflows/ci.yml`):

- **lint** — Ruff + Black on new code
- **test** — `pytest -m "not gpu"` against a PostgreSQL 16 service container
- **docker** — builds both images with layer caching, smoke-tests each

A nightly job (`.github/workflows/integration.yml`) runs the full 12-phase suite on the thesis machine (self-hosted GPU runner) and fails if latency, throughput, or edge FPS regress past the gates in the table above.

---

## Reference

> Z. Parvin, *Real-Time Re-Identification in Distributed Camera Networks*, Master's Thesis, 2025.

The thesis PDF is in this repository.
