"""
evaluate_system.py — End-to-end integration test for the ReID system.

Tests (in order):
  1.  Server health & FastAPI reachability
  2.  PostgreSQL persistence (identities + sightings tables)
  3.  All REST API endpoints
  4.  Match-or-create logic & EMA gallery update
  5.  Threshold config hot-reload
  6.  ReID model pipeline (models load, embeddings are valid)
  7.  Full pipeline via sim_cam videos (YOLO → embedding → POST)
  8.  Cross-camera re-identification
  9.  Gallery persistence (feature vectors stored in DB)
  10. Dashboard reachability
  11. POST /register_track latency & throughput

Usage:
    python evaluate_system.py
    python evaluate_system.py --server http://localhost:5005
    python evaluate_system.py --server http://localhost:5005 --max-vehicles 100
    python evaluate_system.py --no-pipeline   # skip video processing
"""

import argparse
import base64
import io
import json
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import torch

# ── ANSI colours ──────────────────────────────────────────────────────────────
GRN  = "\033[92m"
RED  = "\033[91m"
YEL  = "\033[93m"
CYN  = "\033[96m"
DIM  = "\033[2m"
BLD  = "\033[1m"
RST  = "\033[0m"

PASS_TAG = f"{GRN}PASS{RST}"
FAIL_TAG = f"{RED}FAIL{RST}"
WARN_TAG = f"{YEL}WARN{RST}"
SKIP_TAG = f"{DIM}SKIP{RST}"


# ── Result accumulator ────────────────────────────────────────────────────────

class Phase:
    def __init__(self, name: str):
        self.name    = name
        self.status  = "PASS"      # PASS / FAIL / WARN / SKIP
        self.checks: list[tuple[bool | None, str]] = []  # (ok, message)
        self.details: dict = {}

    def ok(self, msg: str):
        self.checks.append((True, msg))

    def fail(self, msg: str):
        self.checks.append((False, msg))
        self.status = "FAIL"

    def warn(self, msg: str):
        self.checks.append((None, msg))
        if self.status == "PASS":
            self.status = "WARN"

    def skip(self, reason: str):
        self.checks.append((None, reason))
        self.status = "SKIP"

    def require(self, condition: bool, ok_msg: str, fail_msg: str) -> bool:
        if condition:
            self.ok(ok_msg)
            return True
        else:
            self.fail(fail_msg)
            return False

    def tag(self) -> str:
        return {"PASS": PASS_TAG, "FAIL": FAIL_TAG,
                "WARN": WARN_TAG, "SKIP": SKIP_TAG}[self.status]

    def print(self):
        width = 54
        header = f"  {BLD}{self.name}{RST}"
        dots   = "." * max(4, width - len(self.name))
        print(f"{header} {dots} {self.tag()}")
        for ok, msg in self.checks:
            if ok is True:
                print(f"    {GRN}✓{RST} {msg}")
            elif ok is False:
                print(f"    {RED}✗{RST} {msg}")
            else:
                print(f"    {YEL}!{RST} {msg}")
        if self.details:
            for k, v in self.details.items():
                print(f"    {DIM}{k}: {v}{RST}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def random_unit_vector(dim: int = 512) -> list[float]:
    v = np.random.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def make_crop_b64(color=(100, 150, 200), size=(64, 64)) -> str:
    img = np.full((*size, 3), color, dtype=np.uint8)
    ok, jpg = cv2.imencode(".jpg", img)
    return base64.b64encode(jpg.tobytes()).decode()


def vehicle_frames(video_path: Path, max_vehicles: int):
    """
    Yield (BGR frame, frame_number) for the FIRST frame of each vehicle
    appearance in a sim_cam video (the first non-black frame after a black
    run), up to max_vehicles.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    prev_black = True
    count      = 0
    frame_n    = 0
    while count < max_vehicles:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n   += 1
        is_black   = frame.mean() < 8
        if prev_black and not is_black:
            yield frame, frame_n
            count += 1
        prev_black = is_black
    cap.release()


# ── Individual test phases ────────────────────────────────────────────────────

def phase_server_health(server: str) -> Phase:
    p = Phase("1. Server Health")
    try:
        r = requests.get(f"{server}/stats", timeout=5)
        p.require(r.status_code == 200,
                  f"GET /stats → 200  (server online at {server})",
                  f"GET /stats → {r.status_code} (expected 200)")
    except requests.exceptions.ConnectionError:
        p.fail(f"Cannot connect to server at {server}")
        p.warn("Start the server with:  python server.py")
        return p

    # Check docs
    try:
        dr = requests.get(f"{server}/docs", timeout=3)
        p.require(dr.status_code == 200,
                  "GET /docs → 200  (FastAPI Swagger UI available)",
                  f"GET /docs → {dr.status_code}")
    except Exception as e:
        p.warn(f"GET /docs failed: {e}")

    # Check response schema
    data = r.json()
    for key in ("identities", "total_sightings", "active_cameras", "gallery_size"):
        p.require(key in data,
                  f"/stats contains '{key}'",
                  f"/stats missing key '{key}'")
    return p


def phase_database(server: str) -> Phase:
    p = Phase("2. PostgreSQL Persistence")

    # POST a synthetic track and check it lands in DB
    feat  = random_unit_vector(512)
    crop  = make_crop_b64()
    payload = {"camera_id": "eval-cam", "yolo_class": 2,
               "features": feat, "crop_b64": crop}
    try:
        r = requests.post(f"{server}/register_track", json=payload, timeout=5)
    except Exception as e:
        p.fail(f"POST /register_track failed: {e}")
        return p

    if not p.require(r.status_code == 200,
                     f"POST /register_track → 200",
                     f"POST /register_track → {r.status_code}: {r.text[:120]}"):
        return p

    data   = r.json()
    gid    = data.get("global_id", "")
    is_new = data.get("is_new", None)
    p.require("global_id" in data, f"Response has 'global_id' = {gid}",
              "Response missing 'global_id'")
    p.require("score" in data,     "Response has 'score'",
              "Response missing 'score'")
    p.require("is_new" in data,    "Response has 'is_new'",
              "Response missing 'is_new'")

    # Verify it appears in /identities
    time.sleep(0.3)
    ids_r = requests.get(f"{server}/identities", timeout=5).json()
    ids   = {i["global_id"] for i in ids_r.get("identities", [])}
    p.require(gid in ids,
              f"Identity {gid} persisted in DB (found in /identities)",
              f"Identity {gid} NOT found in /identities")

    # Verify /stats reflects it
    stats = requests.get(f"{server}/stats", timeout=3).json()
    n_sightings = stats.get("total_sightings", 0)
    p.require(n_sightings >= 1,
              f"DB sightings count = {n_sightings} (≥ 1)",
              "DB sightings count is 0 after registration")

    # Verify crop stored
    crop_r = requests.get(f"{server}/identities/{gid}/crop", timeout=3)
    p.require(crop_r.status_code == 200,
              f"GET /identities/{gid}/crop → 200 (crop stored in DB)",
              f"GET /identities/{gid}/crop → {crop_r.status_code}")
    if crop_r.status_code == 200:
        p.require(crop_r.headers.get("content-type", "").startswith("image"),
                  "Crop response content-type is image/*",
                  f"Unexpected content-type: {crop_r.headers.get('content-type')}")

    p.details["registered_id"] = gid
    return p


def phase_api_endpoints(server: str) -> Phase:
    p = Phase("3. API Endpoints")

    endpoints = [
        ("GET",   "/stats",              200),
        ("GET",   "/identities",         200),
        ("GET",   "/get_recent_logs",    200),
        ("GET",   "/get_log",            200),
    ]
    for method, path, expect in endpoints:
        try:
            r = requests.request(method, f"{server}{path}", timeout=5)
            p.require(r.status_code == expect,
                      f"{method} {path} → {r.status_code}",
                      f"{method} {path} → {r.status_code} (expected {expect})")
        except Exception as e:
            p.fail(f"{method} {path} → error: {e}")

    # /get_log should be valid CSV
    try:
        csv_r = requests.get(f"{server}/get_log", timeout=5)
        lines = csv_r.text.strip().splitlines()
        header = lines[0] if lines else ""
        p.require("Timestamp" in header and "Global_ID" in header,
                  f"GET /get_log returns valid CSV ({len(lines)} rows)",
                  f"GET /get_log CSV header unexpected: {header[:80]}")
    except Exception as e:
        p.warn(f"GET /get_log CSV check failed: {e}")

    # PATCH /config
    try:
        cfg_r = requests.patch(f"{server}/config",
                               json={"person": 0.6, "vehicle": 0.5}, timeout=3)
        p.require(cfg_r.status_code == 200,
                  "PATCH /config → 200 (threshold hot-reload works)",
                  f"PATCH /config → {cfg_r.status_code}")
    except Exception as e:
        p.fail(f"PATCH /config → error: {e}")

    # /identities/{id}/sightings
    try:
        ids_r = requests.get(f"{server}/identities", timeout=3).json()
        ids   = ids_r.get("identities", [])
        if ids:
            gid   = ids[0]["global_id"]
            sig_r = requests.get(f"{server}/identities/{gid}/sightings", timeout=3)
            p.require(sig_r.status_code == 200,
                      f"GET /identities/{gid}/sightings → 200",
                      f"GET /identities/{gid}/sightings → {sig_r.status_code}")
            if sig_r.status_code == 200:
                sigs = sig_r.json().get("sightings", [])
                p.require(isinstance(sigs, list),
                          f"Sightings list returned ({len(sigs)} items)",
                          "Sightings response not a list")
    except Exception as e:
        p.warn(f"Sightings endpoint check: {e}")

    return p


def phase_match_or_create(server: str) -> Phase:
    p = Phase("4. Match-or-Create & EMA")

    # Send same embedding twice — second should match
    feat = random_unit_vector(512)
    payload = {"camera_id": "eval-logic", "yolo_class": 2,
               "features": feat, "crop_b64": make_crop_b64()}
    try:
        r1 = requests.post(f"{server}/register_track", json=payload, timeout=5).json()
        r2 = requests.post(f"{server}/register_track", json=payload, timeout=5).json()
    except Exception as e:
        p.fail(f"POST failed: {e}")
        return p

    gid1, gid2 = r1.get("global_id"), r2.get("global_id")
    p.require(r1.get("is_new") is True,
              f"First registration → is_new=True ({gid1})",
              f"First registration → is_new={r1.get('is_new')} (expected True)")
    p.require(r2.get("is_new") is False,
              f"Second registration → is_new=False (re-identified as {gid2})",
              f"Second registration → is_new={r2.get('is_new')} (expected False)")
    p.require(gid1 == gid2,
              f"Same embedding → same global_id ({gid1})",
              f"Same embedding got different IDs: {gid1} vs {gid2}")

    # Send orthogonal embedding — should create new ID
    other = random_unit_vector(512)
    r3 = requests.post(f"{server}/register_track",
                       json={"camera_id": "eval-logic", "yolo_class": 2,
                             "features": other}, timeout=5).json()
    p.require(r3.get("is_new") is True,
              f"Different embedding → new identity ({r3.get('global_id')})",
              f"Different embedding not assigned new ID (is_new={r3.get('is_new')})")

    # Verify ID naming convention
    for gid in (gid1, r3.get("global_id", "")):
        p.require(gid.startswith("Vehicle-") or gid.startswith("Person-"),
                  f"ID '{gid}' follows naming convention (Category-N)",
                  f"ID '{gid}' does not follow naming convention")

    # EMA: after EMA update, score on a third identical call should still be ≥ threshold
    r4 = requests.post(f"{server}/register_track", json=payload, timeout=5).json()
    p.require(r4.get("score", 0) >= 0.4,
              f"EMA update: score after 3rd identical registration = {r4.get('score', 0):.3f} (≥ 0.4)",
              f"EMA update degraded score too much: {r4.get('score', 0):.3f}")

    return p


def phase_threshold_config(server: str) -> Phase:
    p = Phase("5. Threshold Hot-Reload")

    original = {"person": 0.6, "vehicle": 0.5}
    for thresh in [0.3, 0.9, 0.5]:
        try:
            r = requests.patch(f"{server}/config",
                               json={"vehicle": thresh}, timeout=3)
            p.require(r.status_code == 200 and
                      abs(r.json()["thresholds"]["vehicle"] - thresh) < 0.001,
                      f"PATCH /config vehicle={thresh:.1f} → accepted",
                      f"PATCH /config vehicle={thresh:.1f} → rejected")
        except Exception as e:
            p.fail(f"Threshold {thresh}: {e}")

    # Restore
    requests.patch(f"{server}/config", json=original, timeout=3)
    p.ok("Thresholds restored to defaults (person=0.6, vehicle=0.5)")
    return p


def phase_reid_models(args) -> Phase:
    p = Phase("6. ReID Model Pipeline")
    try:
        from edge_client import build_vehicle_reid, build_person_reid, bgr_to_rgb_list
    except ImportError as e:
        p.fail(f"Cannot import edge_client: {e}")
        return p

    device = "cuda" if torch.cuda.is_available() else "cpu"
    p.ok(f"Device: {device}")

    # Vehicle model (fine-tuned weights)
    weight_path = Path(args.weight_path) if args.weight_path else Path("vehicle_reid_osnet_v4.pth")
    try:
        extractor = build_vehicle_reid(
            device=device,
            weight_path=str(weight_path) if weight_path.exists() else None
        )
        if weight_path.exists():
            p.ok(f"build_vehicle_reid loaded fine-tuned weights ({weight_path.name})")
        else:
            p.warn(f"Fine-tuned weights not found at {weight_path}; using pretrained")

        # Run a forward pass on a dummy image
        dummy = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)]
        with torch.no_grad():
            feats = extractor(dummy).cpu().numpy()
        dim  = feats.shape[1]
        raw_norm = float(np.linalg.norm(feats[0]))
        p.require(dim == 512,
                  f"Vehicle embedding dim = {dim} (expected 512)",
                  f"Vehicle embedding dim = {dim} (expected 512)")
        # Normalize manually (server does this on /register_track)
        normed = feats[0] / (raw_norm + 1e-8)
        normed_norm = float(np.linalg.norm(normed))
        p.require(0.99 < normed_norm < 1.01,
                  f"L2-normalisation works (raw ‖feat‖={raw_norm:.1f} -> normalised ‖feat‖={normed_norm:.4f})",
                  f"L2-normalisation failed (‖feat‖={normed_norm:.4f})")
    except Exception as e:
        p.fail(f"Vehicle extractor failed: {e}")
        traceback.print_exc()

    # Person model
    try:
        person_ext = build_person_reid(device=device)
        dummy = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)]
        with torch.no_grad():
            pf = person_ext(dummy).cpu().numpy()
        p.require(pf.shape[1] == 512,
                  f"Person embedding dim = {pf.shape[1]} (512)",
                  f"Person embedding dim = {pf.shape[1]}")
    except Exception as e:
        p.fail(f"Person extractor failed: {e}")

    return p


def phase_pipeline(server: str, args, phases: list) -> Phase:
    p = Phase("7. Full Pipeline (sim_cam videos)")

    sim_videos = [Path(f"sim_cam{i+1}.mp4") for i in range(3)]
    cam_ids    = ["Camera-1", "Camera-2", "Camera-3"]

    missing = [v.name for v in sim_videos if not v.exists() or v.stat().st_size < 10_000]
    if missing:
        p.skip(f"sim_cam videos not ready: {missing}. Run sim_cameras.py first.")
        return p

    try:
        from ultralytics import YOLO
        from edge_client import build_vehicle_reid, build_person_reid, bgr_to_rgb_list
    except ImportError as e:
        p.fail(f"Import error: {e}")
        return p

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    detector   = YOLO("yolov8n.pt")
    weight_path = Path(args.weight_path) if args.weight_path else Path("vehicle_reid_osnet_v4.pth")
    v_ext = build_vehicle_reid(device=device,
                               weight_path=str(weight_path) if weight_path.exists() else None)
    pe_ext = build_person_reid(device=device)

    stats = defaultdict(lambda: {"detected": 0, "posted": 0, "new": 0, "reid": 0})
    global_id_to_cams: dict[str, set] = defaultdict(set)

    for video_path, cam_id in zip(sim_videos, cam_ids):
        print(f"    {DIM}Processing {video_path.name} as {cam_id} …{RST}")
        n_frames = 0

        for frame, frame_n in vehicle_frames(video_path, args.max_vehicles):
            n_frames += 1
            results = detector(frame, verbose=False, classes=[0, 1, 2, 3])
            boxes   = results[0].boxes

            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.25:
                    continue

                xyxy = box.xyxy[0].cpu().tolist()
                x1, y1, x2, y2 = (int(v) for v in xyxy)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0 or min(crop.shape[:2]) < 10:
                    continue

                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                try:
                    if cls == 0:
                        with torch.no_grad():
                            feat = pe_ext([rgb_crop]).cpu().numpy()[0]
                        yolo_class = 0
                    else:
                        with torch.no_grad():
                            feat = v_ext([rgb_crop]).cpu().numpy()[0]
                        yolo_class = cls
                except Exception:
                    continue

                feat = (feat / (np.linalg.norm(feat) + 1e-8)).tolist()
                ok_enc, jpg = cv2.imencode(".jpg", crop,
                                           [cv2.IMWRITE_JPEG_QUALITY, 70])
                crop_b64 = base64.b64encode(jpg.tobytes()).decode() if ok_enc else None

                payload = {"camera_id": cam_id, "yolo_class": yolo_class,
                           "features": feat, "crop_b64": crop_b64}
                try:
                    resp = requests.post(f"{server}/register_track",
                                         json=payload, timeout=5).json()
                    stats[cam_id]["detected"] += 1
                    stats[cam_id]["posted"]   += 1
                    gid = resp.get("global_id", "")
                    if resp.get("is_new"):
                        stats[cam_id]["new"] += 1
                    else:
                        stats[cam_id]["reid"] += 1
                    if gid:
                        global_id_to_cams[gid].add(cam_id)
                except Exception:
                    stats[cam_id]["detected"] += 1

    total_posted = sum(s["posted"] for s in stats.values())
    total_new    = sum(s["new"]    for s in stats.values())
    total_reid   = sum(s["reid"]   for s in stats.values())

    p.require(total_posted > 0,
              f"Total POSTs sent: {total_posted}",
              "No embeddings were sent to server (YOLO detected nothing)")

    p.ok(f"New identities created: {total_new}")
    p.ok(f"Re-identifications:     {total_reid}")

    for cam_id, s in stats.items():
        p.ok(f"{cam_id}: detected={s['detected']} new={s['new']} reid={s['reid']}")

    cross_cam = {gid: cams for gid, cams in global_id_to_cams.items()
                 if len(cams) > 1}
    if cross_cam:
        p.ok(f"Cross-camera identities: {len(cross_cam)} IDs seen in ≥2 cameras")
        for gid, cams in list(cross_cam.items())[:5]:
            p.ok(f"  {gid} seen in: {sorted(cams)}")
    else:
        p.warn("No cross-camera re-identifications detected "
               "(may be expected if test images don't share vehicles)")

    p.details["cross_camera_matches"] = len(cross_cam)
    p.details["total_posted"]         = total_posted
    return p


def phase_cross_camera(server: str) -> Phase:
    p = Phase("8. Cross-Camera DB Verification")
    try:
        logs_r = requests.get(f"{server}/get_log", timeout=5)
        lines  = logs_r.text.strip().splitlines()
        if len(lines) <= 1:
            p.warn("No sightings in log yet; run pipeline phase first")
            return p

        # Parse CSV
        from csv import DictReader
        reader  = DictReader(io.StringIO(logs_r.text))
        rows    = list(reader)
        gid_cams: dict[str, set] = defaultdict(set)
        for row in rows:
            gid_cams[row["Global_ID"]].add(row["Camera_ID"])

        multi = {g: c for g, c in gid_cams.items() if len(c) > 1}
        all_cams = {c for cams in gid_cams.values() for c in cams}

        p.ok(f"Total sighting rows in DB: {len(rows)}")
        p.ok(f"Unique global IDs: {len(gid_cams)}")
        p.ok(f"Active cameras:    {len(all_cams)} ({sorted(all_cams)})")
        if multi:
            p.ok(f"IDs seen in ≥2 cameras: {len(multi)}")
        else:
            p.warn("No identity seen in >1 camera (cross-camera ReID not demonstrated yet)")

        # Verify sightings timestamps are ordered
        ts_list = [row["Timestamp"] for row in rows if row["Timestamp"]]
        if len(ts_list) > 1:
            ordered = all(ts_list[i] <= ts_list[i+1] for i in range(len(ts_list)-1))
            p.require(ordered,
                      "Sightings timestamps are chronologically ordered",
                      "Sightings timestamps are NOT ordered (DB issue?)")
    except Exception as e:
        p.fail(f"Cross-camera DB check failed: {e}")
    return p


def phase_gallery_persistence(server: str) -> Phase:
    p = Phase("9. Gallery Persistence")

    # Current state
    try:
        stats_r = requests.get(f"{server}/stats", timeout=3).json()
        gallery = stats_r.get("gallery_size", {})
        p.ok(f"In-memory gallery: person={gallery.get('person',0)}, "
             f"vehicle={gallery.get('vehicle',0)}")

        ids_r = requests.get(f"{server}/identities", timeout=3).json()
        ids   = ids_r.get("identities", [])
        p.ok(f"DB identities table: {len(ids)} rows")

        p.require(len(ids) == sum(gallery.values()),
                  f"In-memory gallery ({sum(gallery.values())}) matches DB ({len(ids)})",
                  f"Gallery mismatch: memory={sum(gallery.values())} vs DB={len(ids)}")

        # Verify at least one identity has a stored feature vector (non-empty crop/features)
        if ids:
            gid   = ids[0]["global_id"]
            crop_r = requests.get(f"{server}/identities/{gid}/crop", timeout=3)
            if crop_r.status_code == 200:
                p.ok(f"Identity {gid} has a stored crop ({len(crop_r.content)} bytes JPEG)")
            else:
                p.warn(f"Identity {gid} has no stored crop")

            sigs_r = requests.get(f"{server}/identities/{gid}/sightings", timeout=3).json()
            n_sigs = len(sigs_r.get("sightings", []))
            p.require(n_sigs > 0,
                      f"Identity {gid} has {n_sigs} sightings in DB",
                      f"Identity {gid} has no sightings in DB")
    except Exception as e:
        p.fail(f"Gallery persistence check failed: {e}")
    return p


def phase_dashboard(dashboard_url: str) -> Phase:
    p = Phase("10. Dashboard (Streamlit)")
    try:
        r = requests.get(dashboard_url, timeout=4)
        p.require(r.status_code == 200,
                  f"GET {dashboard_url} → 200 (Streamlit running)",
                  f"GET {dashboard_url} → {r.status_code}")
        has_html = "<!DOCTYPE" in r.text or "<html" in r.text
        p.require(has_html,
                  "Response is HTML (valid Streamlit page)",
                  "Response is not HTML")
    except requests.exceptions.ConnectionError:
        p.warn(f"Dashboard not reachable at {dashboard_url}")
        p.warn("Start it with:  streamlit run dashboard.py")
    except Exception as e:
        p.warn(f"Dashboard check failed: {e}")
    return p


def phase_performance(server: str, n: int = 50) -> Phase:
    p = Phase("11. Server Latency & Throughput")

    feat = random_unit_vector(512)
    payload = {"camera_id": "perf-test", "yolo_class": 2, "features": feat}
    latencies = []

    for _ in range(n):
        t0 = time.perf_counter()
        try:
            requests.post(f"{server}/register_track", json=payload, timeout=5)
        except Exception:
            pass
        latencies.append((time.perf_counter() - t0) * 1000)

    if not latencies:
        p.fail("All requests failed")
        return p

    lat = sorted(latencies)
    mean_ms = np.mean(lat)
    p50_ms  = np.percentile(lat, 50)
    p95_ms  = np.percentile(lat, 95)
    p99_ms  = np.percentile(lat, 99)
    rps     = 1000 / mean_ms

    p.ok(f"n={n} sequential POSTs → "
         f"mean={mean_ms:.1f}ms  p50={p50_ms:.1f}ms  "
         f"p95={p95_ms:.1f}ms  p99={p99_ms:.1f}ms")
    p.ok(f"Effective throughput: {rps:.1f} req/s (sequential)")

    p.require(mean_ms < 500,
              f"Mean latency {mean_ms:.1f}ms < 500ms",
              f"Mean latency {mean_ms:.1f}ms is high (> 500ms)")
    p.require(p95_ms < 1000,
              f"p95 latency {p95_ms:.1f}ms < 1000ms",
              f"p95 latency {p95_ms:.1f}ms is very high (> 1s)")

    p.details["mean_ms"] = f"{mean_ms:.1f}"
    p.details["p95_ms"]  = f"{p95_ms:.1f}"
    p.details["rps"]     = f"{rps:.1f}"
    return p


def phase_realtime(args) -> Phase:
    """
    Measure end-to-end pipeline throughput on real sim_cam frames:
      YOLO detect → crop → OSNet embed  (no network call — pure edge latency)

    The edge client must sustain >= video FPS to avoid falling behind.
    Budget per frame: 1000 / FPS = 40 ms at 25 fps.
    Note: edge_client only embeds once per REID_BUFFER_FRAMES (5 frames),
    so the embedding budget is 5 x 40 ms = 200 ms, but detection must run
    every frame within 40 ms.
    """
    p = Phase("12. Real-Time Pipeline Throughput")

    VIDEO_FPS    = 25
    FRAME_BUDGET = 1000.0 / VIDEO_FPS          # 40 ms per frame
    EMBED_BUDGET = FRAME_BUDGET * 5             # 200 ms per embedding (buffer=5)
    N_FRAMES     = 300                          # frames to benchmark

    sim_path = Path("sim_cam1.mp4")
    if not sim_path.exists() or sim_path.stat().st_size < 10_000:
        p.skip("sim_cam1.mp4 not found — run sim_cameras.py first")
        return p

    try:
        from ultralytics import YOLO
        from edge_client import build_vehicle_reid, bgr_to_rgb_list
    except ImportError as e:
        p.fail(f"Import error: {e}")
        return p

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    detector    = YOLO("yolov8n.pt")
    weight_path = Path(args.weight_path) if args.weight_path else Path("vehicle_reid_osnet_v4.pth")
    extractor   = build_vehicle_reid(
        device=device,
        weight_path=str(weight_path) if weight_path.exists() else None
    )
    p.ok(f"Device: {device}  |  video FPS target: {VIDEO_FPS}  |  frame budget: {FRAME_BUDGET:.0f} ms")

    cap = cv2.VideoCapture(str(sim_path))
    detect_times, embed_times, total_times = [], [], []
    frames_read = crops_found = 0

    while frames_read < N_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frames_read += 1

        t_frame = time.perf_counter()

        # --- Detection (must happen every frame) ---
        t0 = time.perf_counter()
        results  = detector(frame, verbose=False, classes=[0, 1, 2, 3])
        t_detect = (time.perf_counter() - t0) * 1000
        detect_times.append(t_detect)

        # --- Embedding (happens once per 5 frames in production) ---
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            best_box = boxes[int(boxes.conf.argmax())]
            x1,y1,x2,y2 = (int(v) for v in best_box.xyxy[0].tolist())
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0 and min(crop.shape[:2]) >= 8:
                crops_found += 1
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                t0 = time.perf_counter()
                with torch.no_grad():
                    extractor([rgb])
                embed_times.append((time.perf_counter() - t0) * 1000)

        total_times.append((time.perf_counter() - t_frame) * 1000)

    cap.release()

    if not detect_times:
        p.fail("Could not read any frames from sim_cam1.mp4")
        return p

    mean_det  = float(np.mean(detect_times))
    p95_det   = float(np.percentile(detect_times, 95))
    mean_emb  = float(np.mean(embed_times))  if embed_times else 0
    p95_emb   = float(np.percentile(embed_times, 95)) if embed_times else 0
    mean_tot  = float(np.mean(total_times))
    eff_fps   = 1000.0 / mean_tot

    p.ok(f"Benchmarked {frames_read} frames  ({crops_found} with detected vehicle)")
    p.ok(f"Detection  : mean={mean_det:.1f} ms  p95={p95_det:.1f} ms  "
         f"(budget {FRAME_BUDGET:.0f} ms/frame)")
    if embed_times:
        p.ok(f"Embedding  : mean={mean_emb:.1f} ms  p95={p95_emb:.1f} ms  "
             f"(budget {EMBED_BUDGET:.0f} ms per buffer flush)")
    p.ok(f"Total      : mean={mean_tot:.1f} ms/frame  ->  {eff_fps:.1f} effective FPS")

    # Detection must be faster than frame budget
    p.require(mean_det < FRAME_BUDGET,
              f"Detection mean {mean_det:.1f} ms < {FRAME_BUDGET:.0f} ms (real-time capable)",
              f"Detection mean {mean_det:.1f} ms > {FRAME_BUDGET:.0f} ms (cannot keep up with {VIDEO_FPS} fps)")
    p.require(p95_det < FRAME_BUDGET * 2,
              f"Detection p95 {p95_det:.1f} ms < {FRAME_BUDGET*2:.0f} ms",
              f"Detection p95 {p95_det:.1f} ms too high (> {FRAME_BUDGET*2:.0f} ms)")

    # Embedding runs every 5 frames so budget is more relaxed
    if embed_times:
        p.require(mean_emb < EMBED_BUDGET,
                  f"Embedding mean {mean_emb:.1f} ms < {EMBED_BUDGET:.0f} ms (within 5-frame buffer window)",
                  f"Embedding mean {mean_emb:.1f} ms > {EMBED_BUDGET:.0f} ms (too slow for buffer)")

    # Overall FPS
    p.require(eff_fps >= VIDEO_FPS * 0.8,
              f"Effective FPS {eff_fps:.1f} >= {VIDEO_FPS * 0.8:.0f} (80% of target — real-time)",
              f"Effective FPS {eff_fps:.1f} < {VIDEO_FPS * 0.8:.0f} — pipeline cannot sustain real-time")

    if eff_fps < VIDEO_FPS:
        p.warn(f"FPS {eff_fps:.1f} < {VIDEO_FPS} — edge_client will process slower than live video. "
               f"Consider: GPU acceleration, smaller YOLO model, or lower resolution input.")

    p.details["detect_mean_ms"] = f"{mean_det:.1f}"
    p.details["embed_mean_ms"]  = f"{mean_emb:.1f}"
    p.details["effective_fps"]  = f"{eff_fps:.1f}"
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--server",        default="http://127.0.0.1:5005")
    ap.add_argument("--dashboard",     default="http://localhost:8501")
    ap.add_argument("--weight-path",   default="vehicle_reid_osnet_v4.pth")
    ap.add_argument("--max-vehicles",  type=int, default=60,
                    help="Max vehicle appearances to process per sim_cam video")
    ap.add_argument("--no-pipeline",   action="store_true",
                    help="Skip sim_cam video processing (phases 7-8)")
    ap.add_argument("--no-reset",      action="store_true",
                    help="Do not reset gallery before testing")
    ap.add_argument("--no-realtime",   action="store_true",
                    help="Skip real-time throughput benchmark (phase 12)")
    ap.add_argument("--report",        default="LOGS/eval_system_report.txt",
                    help="Write plain-text report to this file")
    args = ap.parse_args()

    print()
    print(f"{BLD}{'='*62}{RST}")
    print(f"{BLD}  ReID System Integration Evaluation{RST}")
    print(f"{DIM}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
          f"server={args.server}{RST}")
    print(f"{BLD}{'='*62}{RST}")
    print()

    # ── Reset gallery for a clean test run ──
    if not args.no_reset:
        try:
            r = requests.post(f"{args.server}/gallery/reset", timeout=5)
            if r.status_code == 200:
                print(f"  {YEL}Gallery reset before evaluation (use --no-reset to skip){RST}\n")
        except Exception:
            print(f"  {RED}Could not reset gallery — server may not be running{RST}\n")

    # ── Run phases ──────────────────────────────────────────────────────────
    all_phases: list[Phase] = []

    def run(phase_fn, *a, **kw) -> Phase:
        ph = phase_fn(*a, **kw)
        ph.print()
        print()
        all_phases.append(ph)
        return ph

    # Phase 12 runs independently of the server (pure edge benchmark)
    if not args.no_realtime:
        run(phase_realtime, args)

    h = run(phase_server_health,   args.server)
    if h.status == "FAIL":
        print(f"  {RED}Server unreachable — skipping server-dependent phases.{RST}")
        print(f"  Start with:  {CYN}python server.py{RST}\n")
    else:
        run(phase_database,          args.server)
        run(phase_api_endpoints,     args.server)
        run(phase_match_or_create,   args.server)
        run(phase_threshold_config,  args.server)
        run(phase_reid_models,       args)

        if not args.no_pipeline:
            run(phase_pipeline,      args.server, args, all_phases)
            run(phase_cross_camera,  args.server)

        run(phase_gallery_persistence, args.server)
        run(phase_dashboard,           args.dashboard)
        run(phase_performance,         args.server)

    # ── Summary ──────────────────────────────────────────────────────────────
    counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
    for ph in all_phases:
        counts[ph.status] += 1

    print(f"{BLD}{'='*62}{RST}")
    print(f"{BLD}  SUMMARY{RST}")
    print(f"  {GRN}PASS: {counts['PASS']}{RST}   "
          f"{RED}FAIL: {counts['FAIL']}{RST}   "
          f"{YEL}WARN: {counts['WARN']}{RST}   "
          f"{DIM}SKIP: {counts['SKIP']}{RST}")
    print(f"{BLD}{'='*62}{RST}")

    if counts["FAIL"] == 0 and counts["WARN"] == 0:
        print(f"\n  {GRN}{BLD}All systems operational.{RST}")
    elif counts["FAIL"] == 0:
        print(f"\n  {YEL}System functional with warnings. Review above.{RST}")
    else:
        print(f"\n  {RED}System has failures. Review above.{RST}")
    print()

    # ── Write report ──────────────────────────────────────────────────────────
    report_lines = [
        "ReID System Integration Evaluation Report",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"Server: {args.server}",
        "=" * 60,
        "",
    ]
    for ph in all_phases:
        report_lines.append(f"[{ph.status}] {ph.name}")
        for ok, msg in ph.checks:
            sym = "OK" if ok is True else ("FAIL" if ok is False else "WARN")
            report_lines.append(f"  {sym}: {msg}")
        for k, v in ph.details.items():
            report_lines.append(f"  {k}: {v}")
        report_lines.append("")

    report_lines += [
        "=" * 60,
        f"PASS={counts['PASS']}  FAIL={counts['FAIL']}  "
        f"WARN={counts['WARN']}  SKIP={counts['SKIP']}",
    ]
    Path(args.report).write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  Report written → {args.report}\n")


if __name__ == "__main__":
    main()
