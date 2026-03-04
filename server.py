import os
from flask import Flask, request, jsonify, send_file
import numpy as np
import torch
import torch.nn.functional as F
import psycopg2
from psycopg2 import pool
import time
from threading import Lock
from collections import deque

# --- Configuration ---
app = Flask(__name__)
PORT = 5005

# PostgreSQL connection config — adjust to your setup
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "reid_logs",
    "user":     "postgres",
    "password": "ilovebanana"
}

# Use a connection pool for thread safety (replaces log_lock + file open)
db_pool = psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=10, **DB_CONFIG)

def initialize_log():
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS camera_log (
                    camera_log_uuid  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    timestamp        TIMESTAMP NOT NULL,
                    camera_id        VARCHAR(50),
                    global_id        VARCHAR(50),
                    class            VARCHAR(50),
                    score            FLOAT
                );
            """)
        conn.commit()
        print("[Server] PostgreSQL table ready.")
    finally:
        db_pool.putconn(conn)


def log_to_db(timestamp, camera_id, global_id, class_name, score):
    """Insert one detection event into PostgreSQL."""
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO camera_log (timestamp, camera_id, global_id, class, score)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (timestamp, camera_id, global_id, class_name, float(score))
            )
        conn.commit()
    finally:
        db_pool.putconn(conn)




gallery_db = {
    "person": [],
    "vehicle": []
}

id_counters = {
    "person": 0,
    "vehicle": 0
}

THRESHOLDS = {
    "person": 0.6,
    "vehicle": 0.5
}

CLASS_MAP = {
    0: "person",
    1: "vehicle",
    2: "vehicle",
    3: "vehicle"
}

# --- Helper Functions (keep your existing ones) ---
def cosine_sim(A, B):
    A_tensor = torch.tensor(A)
    B_tensor = torch.tensor(B)
    return (A_tensor @ B_tensor.T).numpy()

def find_best_match(features_new, gallery_key):
    threshold = THRESHOLDS[gallery_key]
    gallery = gallery_db[gallery_key]
    
    if len(gallery) == 0:
        return None, 0.0
    
    dim = features_new.shape[1]
    compatible_gallery = [item for item in gallery if item['features'].shape[1] == dim]
    
    if not compatible_gallery:
        return None, 0.0
    
    gallery_feats_matrix = np.vstack([item['features'] for item in compatible_gallery])
    sim_scores = cosine_sim(features_new, gallery_feats_matrix)
    
    best_match_idx = np.argmax(sim_scores[0])
    best_score = sim_scores[0][best_match_idx]
    
    if best_score >= threshold:
        global_id = compatible_gallery[best_match_idx]['global_id']
        return global_id, best_score
    else:
        return None, best_score

def add_to_gallery(features_new, gallery_key):
    id_counters[gallery_key] += 1
    new_global_id = f"{gallery_key.capitalize()}-{id_counters[gallery_key]}"
    
    gallery_db[gallery_key].append({
        'global_id': new_global_id,
        'features': features_new
    })
    
    print(f"[Server] New object registered: {new_global_id}")
    return new_global_id

# ---  API Endpoint with Logging ---
@app.route('/register_track', methods=['POST'])
def register_track():
    data = request.json
    yolo_class = int(data['yolo_class'])
    features_list = data['features']
    camera_id = data.get('camera_id', 'unknown')
    features_new = np.array(features_list, dtype=np.float32).reshape(1, -1)

    gallery_key = CLASS_MAP.get(yolo_class)
    if gallery_key is None:
        return jsonify({"error": "Invalid class"}), 400

    global_id, score = find_best_match(features_new, gallery_key)
    if global_id is None:
        global_id = add_to_gallery(features_new, gallery_key)

    # Log to PostgreSQL instead of CSV
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_to_db(timestamp, camera_id, global_id, gallery_key.capitalize(), score)

    print(f"[Server] Request from [Cam: {camera_id}]. Assigned Global_ID: {global_id} (Score: {score:.3f})")
    return jsonify({"global_id": global_id, "score": float(score)})


# --- New /get_recent_logs endpoint (reads from DB) ---
@app.route('/get_recent_logs', methods=['GET'])
def get_recent_logs():
    lines = int(request.args.get('lines', 100))
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT timestamp, camera_id, global_id, class, score "
                "FROM camera_log ORDER BY timestamp DESC LIMIT %s",
                (lines,)
            )
            rows = cur.fetchall()
    finally:
        db_pool.putconn(conn)

    logs = [f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}" for r in reversed(rows)]
    return jsonify({"logs": logs})


# --- New /get_log endpoint (download as CSV from DB) ---
@app.route('/get_log', methods=['GET'])
def get_log():
    import csv, io
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT timestamp, camera_id, global_id, class, score "
                "FROM camera_log ORDER BY timestamp ASC"
            )
            rows = cur.fetchall()
    finally:
        db_pool.putconn(conn)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Timestamp', 'Camera_ID', 'Global_ID', 'Class', 'Score'])
    writer.writerows(rows)
    output.seek(0)

    from flask import Response
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment; filename=camera_log.csv"}
    )


if __name__ == "__main__":
    print(f"--- ReID Server starting on http://0.0.0.0:{PORT} ---")
    initialize_log()
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)