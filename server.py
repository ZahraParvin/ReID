import io
import base64
from typing import List, Optional
from datetime import datetime
from threading import Lock

import numpy as np
import psycopg2
from psycopg2 import pool
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
import uvicorn

# --- Configuration ---
PORT = 5005

DB_CONFIG = {
    "host":     "127.0.0.1",
    "port":     5432,
    "dbname":   "reid_logs",
    "user":     "postgres",
    "password": "*****"
}

THRESHOLDS        = {"person": 0.6, "vehicle": 0.5}
GALLERY_EMA_ALPHA = 0.9
CLASS_MAP         = {0: "person", 1: "vehicle", 2: "vehicle", 3: "vehicle"}

app     = FastAPI(title="ReID Server", version="2.0",
                  description="Multi-camera Re-Identification central server")
db_pool = psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=10, **DB_CONFIG)

gallery_db   = {"person": [], "vehicle": []}
id_counters  = {"person": 0, "vehicle": 0}
gallery_lock = Lock()
# In-memory latest frame per camera (JPEG bytes)
last_frames = {}
frames_lock = Lock()


# ── Request model ──────────────────────────────────────────────────────────────

class TrackPayload(BaseModel):
    camera_id:  str
    yolo_class: int
    features:   List[float]
    crop_b64:   Optional[str] = None   # base64-encoded JPEG crop

class ThresholdConfig(BaseModel):
    person:  Optional[float] = None
    vehicle: Optional[float] = None


# ── Database setup ─────────────────────────────────────────────────────────────

def initialize_db():
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

            # One row per unique identity ever discovered
            cur.execute("""
                CREATE TABLE IF NOT EXISTS identities (
                    global_id        VARCHAR(50)  PRIMARY KEY,
                    category         VARCHAR(20)  NOT NULL,
                    first_seen_at    TIMESTAMP    NOT NULL,
                    last_seen_at     TIMESTAMP    NOT NULL,
                    total_sightings  INT          NOT NULL DEFAULT 0,
                    best_crop        BYTEA,
                    feature_vector   BYTEA        NOT NULL
                );
            """)

            # One row per detection / re-identification event
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sightings (
                    sighting_id      UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
                    timestamp        TIMESTAMP    NOT NULL,
                    camera_id        VARCHAR(50)  NOT NULL,
                    global_id        VARCHAR(50)  NOT NULL REFERENCES identities(global_id),
                    category         VARCHAR(20)  NOT NULL,
                    match_score      FLOAT        NOT NULL,
                    is_new_identity  BOOLEAN      NOT NULL DEFAULT FALSE,
                    crop_image       BYTEA
                );
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_sight_global ON sightings(global_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sight_camera ON sightings(camera_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sight_ts     ON sightings(timestamp);")

        conn.commit()
        print("[Server] PostgreSQL tables ready.")
    finally:
        db_pool.putconn(conn)


# ── Gallery persistence ────────────────────────────────────────────────────────

def load_gallery_from_db():
    """Rebuild in-memory gallery from identities table on startup."""
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT global_id, category, feature_vector FROM identities")
            rows = cur.fetchall()
    finally:
        db_pool.putconn(conn)

    for global_id, category, feat_bytes in rows:
        feat = np.frombuffer(bytes(feat_bytes), dtype=np.float32).reshape(1, -1)
        gallery_db[category].append({"global_id": global_id, "features": feat})
        n = int(global_id.split("-")[-1])
        if n > id_counters[category]:
            id_counters[category] = n

    total = sum(len(v) for v in gallery_db.values())
    print(f"[Server] Gallery restored: {total} identities.")


def _persist_track(global_id, category, features, crop_bytes, is_new, ts,
                   camera_id, score):
    """Write identity upsert + sighting insert in a single transaction."""
    feat_bytes = features.astype(np.float32).flatten().tobytes()
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL synchronous_commit = OFF")
            if is_new:
                cur.execute("""
                    INSERT INTO identities
                        (global_id, category, first_seen_at, last_seen_at,
                         total_sightings, best_crop, feature_vector)
                    VALUES (%s, %s, %s, %s, 1, %s, %s)
                """, (global_id, category, ts, ts, crop_bytes, feat_bytes))
            else:
                cur.execute("""
                    UPDATE identities SET
                        last_seen_at    = %s,
                        total_sightings = total_sightings + 1,
                        feature_vector  = %s,
                        best_crop       = COALESCE(%s, best_crop)
                    WHERE global_id = %s
                """, (ts, feat_bytes, crop_bytes, global_id))
            cur.execute("""
                INSERT INTO sightings
                    (timestamp, camera_id, global_id, category,
                     match_score, is_new_identity, crop_image)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (ts, camera_id, global_id, category, float(score), is_new, crop_bytes))
        conn.commit()
    finally:
        db_pool.putconn(conn)


# ── Matching ───────────────────────────────────────────────────────────────────

def _find_best_match(features, gallery_key):
    gallery   = gallery_db[gallery_key]
    threshold = THRESHOLDS[gallery_key]
    if not gallery:
        return None, 0.0, -1
    dim    = features.shape[1]
    compat = [i for i, e in enumerate(gallery) if e["features"].shape[1] == dim]
    if not compat:
        return None, 0.0, -1
    G     = np.vstack([gallery[i]["features"] for i in compat])
    sims  = (features @ G.T)[0]
    best  = int(np.argmax(sims))
    score = float(sims[best])
    if score >= threshold:
        return gallery[compat[best]]["global_id"], score, compat[best]
    return None, score, -1


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/register_track")
def register_track(payload: TrackPayload):
    category = CLASS_MAP.get(payload.yolo_class)
    if category is None:
        raise HTTPException(status_code=400, detail="Invalid yolo_class")

    features   = np.array(payload.features, dtype=np.float32).reshape(1, -1)
    feat_norm  = np.linalg.norm(features, axis=1, keepdims=True)
    features   = features / np.maximum(feat_norm, 1e-8)   # cosine similarity needs unit vectors
    crop_bytes = base64.b64decode(payload.crop_b64) if payload.crop_b64 else None
    ts         = datetime.now()

    with gallery_lock:
        global_id, score, match_idx = _find_best_match(features, category)
        is_new = global_id is None

        if is_new:
            id_counters[category] += 1
            global_id = f"{category.capitalize()}-{id_counters[category]}"
            gallery_db[category].append({"global_id": global_id, "features": features})
            saved_feats = features
        else:
            entry = gallery_db[category][match_idx]
            updated           = GALLERY_EMA_ALPHA * entry["features"] + (1 - GALLERY_EMA_ALPHA) * features
            norm              = np.linalg.norm(updated, axis=1, keepdims=True)
            entry["features"] = updated / np.maximum(norm, 1e-8)
            saved_feats       = entry["features"]

    _persist_track(global_id, category, saved_feats, crop_bytes, is_new, ts,
                   payload.camera_id, score)

    print(f"[Server] [{payload.camera_id}] {'NEW' if is_new else 're-id'} -> {global_id}  (score={score:.3f})")
    return {"global_id": global_id, "score": float(score), "is_new": is_new}


@app.get("/stats")
def get_stats():
    """Summary counts for the current session."""
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT category, COUNT(*) FROM identities GROUP BY category")
            identity_counts = dict(cur.fetchall())
            cur.execute("SELECT COUNT(*) FROM sightings")
            total_sightings = cur.fetchone()[0]
    finally:
        db_pool.putconn(conn)

    with frames_lock:
        active_cameras = len(last_frames)

    return {
        "identities":      identity_counts,
        "total_sightings": total_sightings,
        "active_cameras":  active_cameras,
        "gallery_size":    {k: len(v) for k, v in gallery_db.items()},
    }


@app.get("/identities")
def list_identities(category: Optional[str] = None):
    """List all known identities, optionally filtered by category."""
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            if category:
                cur.execute(
                    "SELECT global_id, category, first_seen_at, last_seen_at, total_sightings "
                    "FROM identities WHERE category=%s ORDER BY first_seen_at",
                    (category,)
                )
            else:
                cur.execute(
                    "SELECT global_id, category, first_seen_at, last_seen_at, total_sightings "
                    "FROM identities ORDER BY first_seen_at"
                )
            rows = cur.fetchall()
    finally:
        db_pool.putconn(conn)
    return {"identities": [
        {"global_id": r[0], "category": r[1],
         "first_seen": str(r[2]), "last_seen": str(r[3]), "sightings": r[4]}
        for r in rows
    ]}


@app.get("/identities/{global_id}/crop")
def get_identity_crop(global_id: str):
    """Return the best stored crop image for an identity as JPEG."""
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT best_crop FROM identities WHERE global_id=%s", (global_id,))
            row = cur.fetchone()
    finally:
        db_pool.putconn(conn)
    if not row or row[0] is None:
        raise HTTPException(status_code=404, detail="No crop available for this identity")
    return Response(content=bytes(row[0]), media_type="image/jpeg")


@app.post("/camera/{camera_id}/frame")
def push_camera_frame(camera_id: str, payload: dict):
    """Receive a base64-encoded JPEG frame for a camera and store latest frame in memory."""
    fb64 = payload.get("frame_b64")
    if not fb64:
        raise HTTPException(status_code=400, detail="Missing frame_b64")
    try:
        b = base64.b64decode(fb64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64")
    with frames_lock:
        last_frames[camera_id] = b
    return {"status": "ok"}


@app.get("/camera/{camera_id}/frame")
def get_camera_frame(camera_id: str):
    """Return the latest JPEG frame for a camera, if available."""
    with frames_lock:
        b = last_frames.get(camera_id)
    if not b:
        raise HTTPException(status_code=404, detail="No frame for this camera")
    return Response(content=bytes(b), media_type="image/jpeg")


@app.get("/cameras")
def list_cameras():
    """Return a list of camera IDs that have pushed frames."""
    with frames_lock:
        cams = list(last_frames.keys())
    return {"cameras": cams}


@app.get("/identities/{global_id}/sightings")
def get_identity_sightings(global_id: str, limit: int = 50):
    """Return all sighting events for a given identity."""
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT sighting_id, timestamp, camera_id, match_score, is_new_identity "
                "FROM sightings WHERE global_id=%s ORDER BY timestamp DESC LIMIT %s",
                (global_id, limit)
            )
            rows = cur.fetchall()
    finally:
        db_pool.putconn(conn)
    return {"global_id": global_id, "sightings": [
        {"id": str(r[0]), "timestamp": str(r[1]), "camera_id": r[2],
         "score": r[3], "is_new": r[4]}
        for r in rows
    ]}


@app.get("/sightings/{sighting_id}/crop")
def get_sighting_crop(sighting_id: str):
    """Return the crop image for a specific sighting event."""
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT crop_image FROM sightings WHERE sighting_id=%s::uuid", (sighting_id,))
            row = cur.fetchone()
    finally:
        db_pool.putconn(conn)
    if not row or row[0] is None:
        raise HTTPException(status_code=404, detail="No crop for this sighting")
    return Response(content=bytes(row[0]), media_type="image/jpeg")


@app.patch("/config")
def update_config(cfg: ThresholdConfig):
    """Adjust matching thresholds at runtime."""
    if cfg.person  is not None: THRESHOLDS["person"]  = cfg.person
    if cfg.vehicle is not None: THRESHOLDS["vehicle"] = cfg.vehicle
    return {"status": "ok", "thresholds": THRESHOLDS}


@app.post("/gallery/reset")
def reset_gallery():
    """Clear in-memory gallery and all DB records."""
    with gallery_lock:
        gallery_db["person"].clear()
        gallery_db["vehicle"].clear()
        id_counters["person"]  = 0
        id_counters["vehicle"] = 0
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sightings")
            cur.execute("DELETE FROM identities")
        conn.commit()
    finally:
        db_pool.putconn(conn)
    print("[Server] Gallery reset.")
    return {"status": "ok"}


@app.get("/get_recent_logs")
def get_recent_logs(lines: int = 100):
    """Legacy endpoint — compatible with logger.py."""
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT timestamp, camera_id, global_id, category, match_score "
                "FROM sightings ORDER BY timestamp DESC LIMIT %s",
                (lines,)
            )
            rows = cur.fetchall()
    finally:
        db_pool.putconn(conn)
    logs = [f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}" for r in reversed(rows)]
    return {"logs": logs}


@app.get("/get_log")
def get_log():
    """Download full sightings log as CSV."""
    import csv
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT timestamp, camera_id, global_id, category, match_score, is_new_identity "
                "FROM sightings ORDER BY timestamp ASC"
            )
            rows = cur.fetchall()
    finally:
        db_pool.putconn(conn)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Timestamp", "Camera_ID", "Global_ID", "Category", "Score", "Is_New"])
    writer.writerows(rows)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=camera_log.csv"}
    )


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    print(f"[Server] Starting  ->  http://0.0.0.0:{PORT}")
    print(f"[Server] API docs  ->  http://0.0.0.0:{PORT}/docs")
    initialize_db()
    load_gallery_from_db()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
