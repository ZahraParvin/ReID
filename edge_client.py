import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torchreid
from ultralytics import YOLO
import base64
import requests
import time
from pathlib import Path

_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    """Patch torch.load to use weights_only=False for compatibility with older checkpoints."""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)

torch.load = _patched_torch_load


# Map YOLO class indices to groups for ReID
CLASS_PERSON = 0
CLASS_BICYCLE = 1
CLASS_CAR = 2
CLASS_MOTORCYCLE = 3
VEHICLE_CLASSES = {CLASS_BICYCLE, CLASS_CAR, CLASS_MOTORCYCLE}
REID_BUFFER_FRAMES = 5  # collect this many frames per track, then register the highest-confidence crop


def build_person_reid(device="cpu"):
    """OSNet-AIN-x1.0 for person ReID (Market-1501 / MSMT17 pretrained)."""
    return torchreid.utils.FeatureExtractor(
        model_name="osnet_ain_x1_0",
        image_size=(256, 256),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        device=device,
    )


def build_vehicle_reid(device="cpu", weight_path=None):
    """OSNet-AIN-x1.0 for vehicle ReID, optionally with fine-tuned weights."""
    # Prefer an explicit weight_path; otherwise fall back to a bundled
    # `vehicle_reid_osnet_v4.pth` located next to this file (repo root).
    if weight_path:
        model_path = str(weight_path)
    else:
        default_path = Path(__file__).parent / "vehicle_reid_osnet_v4.pth"
        model_path = str(default_path) if default_path.exists() else ""

    try:
        return torchreid.utils.FeatureExtractor(
            model_name="osnet_ain_x1_0",
            model_path=model_path,
            image_size=(256, 256),
            pixel_mean=[0.485, 0.456, 0.406],
            pixel_std=[0.229, 0.224, 0.225],
            device=device,
        )
    except Exception as e:
        print(f"Warning: Failed to load vehicle weights from '{model_path}': {e}")
        print("Falling back to pretrained OSNet weights.")
        return torchreid.utils.FeatureExtractor(
            model_name="osnet_ain_x1_0",
            image_size=(256, 256),
            pixel_mean=[0.485, 0.456, 0.406],
            pixel_std=[0.229, 0.224, 0.225],
            device=device,
        )


def bgr_to_rgb_list(imgs_bgr):
    """Convert a list of BGR crops to RGB for torchreid."""
    return [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs_bgr]


def crop_xyxy(img, xyxy):
    """Safely crops an image using xyxy coordinates."""
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


# ── Per-frame helpers ──────────────────────────────────────────────────────────

def _push_frame(server_base, camera_id, frame, last_frame_send):
    """Push a full frame to the server for dashboard live-view (throttled to 0.5s)."""
    now = time.time()
    if now - last_frame_send < 0.5:
        return last_frame_send
    ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        try:
            requests.post(f"{server_base}/camera/{camera_id}/frame",
                          json={"frame_b64": base64.b64encode(jpg).decode()},
                          timeout=0.8)
        except Exception:
            pass
    return now


def _update_pending(pending_tracks, track_id, conf, crop, yolo_class):
    """Add a new track to the pending buffer or update it if conf improved."""
    if track_id not in pending_tracks:
        pending_tracks[track_id] = {
            "best_conf": conf, "best_crop": crop,
            "yolo_class": yolo_class, "frames_seen": 1,
        }
    else:
        pending_tracks[track_id]["frames_seen"] += 1
        if conf > pending_tracks[track_id]["best_conf"]:
            pending_tracks[track_id]["best_conf"] = conf
            pending_tracks[track_id]["best_crop"] = crop


def _flush_ready_tracks(pending_tracks, seen_track_ids, current_track_ids,
                        save_crops, crops_dir, camera_id):
    """
    Move tracks whose buffer is full (or that have disappeared) out of pending.
    Returns (person_crops, person_track_ids, vehicle_crops, vehicle_track_ids).
    """
    person_crops, person_track_ids = [], []
    vehicle_crops, vehicle_track_ids = [], []

    for track_id in list(pending_tracks):
        data = pending_tracks[track_id]
        if data["frames_seen"] < REID_BUFFER_FRAMES and track_id in current_track_ids:
            continue
        pending_tracks.pop(track_id)
        seen_track_ids.add(track_id)

        if save_crops:
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(crops_dir, f"{camera_id}_track{track_id}_{ts}.jpg"),
                        data["best_crop"])

        if data["yolo_class"] == CLASS_PERSON:
            person_crops.append(data["best_crop"])
            person_track_ids.append(track_id)
        elif data["yolo_class"] in VEHICLE_CLASSES:
            vehicle_crops.append(data["best_crop"])
            vehicle_track_ids.append((track_id, data["yolo_class"]))

    return person_crops, person_track_ids, vehicle_crops, vehicle_track_ids


def _register_persons(extractor, crops, track_ids, camera_id, register_url,
                      track_id_to_global_id, seen_track_ids):
    """Extract person embeddings and POST each to the server."""
    with torch.no_grad():
        features = extractor(bgr_to_rgb_list(crops))
    for track_id, feat_vector, crop_bgr in zip(track_ids, features, crops):
        try:
            _, jpg = cv2.imencode('.jpg', crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            payload = {
                "camera_id": camera_id,
                "yolo_class": CLASS_PERSON,
                "features": feat_vector.tolist(),
                "crop_b64": base64.b64encode(jpg).decode(),
            }
            response = requests.post(register_url, json=payload, timeout=5)
            if response.status_code == 200:
                global_id = response.json().get("global_id")
                track_id_to_global_id[track_id] = global_id
                print(f"[Client] Person Track {track_id} -> {global_id}")
        except Exception as e:
            print(f"Error connecting to server: {e}")
            seen_track_ids.discard(track_id)  # allow retry on next appearance


def _register_vehicles(extractor, crops, track_ids_and_classes, camera_id, register_url,
                       track_id_to_global_id, seen_track_ids):
    """Extract vehicle embeddings and POST each to the server."""
    with torch.no_grad():
        features = extractor(bgr_to_rgb_list(crops))
    for (track_id, yolo_class), feat_vector, crop_bgr in zip(track_ids_and_classes, features, crops):
        try:
            _, jpg = cv2.imencode('.jpg', crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            payload = {
                "camera_id": camera_id,
                "yolo_class": yolo_class,
                "features": feat_vector.tolist(),
                "crop_b64": base64.b64encode(jpg).decode(),
            }
            response = requests.post(register_url, json=payload, timeout=5)
            if response.status_code == 200:
                global_id = response.json().get("global_id")
                track_id_to_global_id[track_id] = global_id
                print(f"[Client] Vehicle Track {track_id} -> {global_id}")
        except Exception as e:
            print(f"Error connecting to server: {e}")
            seen_track_ids.discard(track_id)  # allow retry on next appearance


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    parser = argparse.ArgumentParser(description="ReID Edge Client")
    parser.add_argument("--camera-id",    default="Camera-1")
    parser.add_argument("--video-source", default="sim_cam1.mp4")
    parser.add_argument("--server-url",   default="http://127.0.0.1:5005/register_track")
    parser.add_argument("--save-crops",   action="store_true",
                        help="Save the best crop per track to disk for ground-truth annotation")
    parser.add_argument("--crops-dir",    default="crops",
                        help="Directory to write crop images (default: crops)")
    parser.add_argument("--weight-path",  default=None,
                        help="Path to fine-tuned OSNet vehicle checkpoint (.pth)")
    args = parser.parse_args()

    camera_id    = args.camera_id
    video_source = args.video_source
    save_crops   = args.save_crops
    crops_dir    = args.crops_dir

    server_base  = args.server_url.rstrip('/')
    if server_base.endswith('/register_track'):
        server_base = server_base[:-len('/register_track')]
    register_url = server_base + '/register_track'

    if save_crops:
        os.makedirs(crops_dir, exist_ok=True)
        print(f"[Client] Saving crops to '{crops_dir}/'")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Client] device={device}")

    detector          = YOLO("yolov8n.pt")
    person_extractor  = build_person_reid(device=device)
    vehicle_extractor = build_vehicle_reid(device=device, weight_path=args.weight_path)

    cap = cv2.VideoCapture(video_source)

    seen_track_ids        = set()
    track_id_to_global_id = {}
    pending_tracks        = {}
    last_frame_send       = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        last_frame_send = _push_frame(server_base, camera_id, frame, last_frame_send)

        results = detector.track(frame, persist=True)
        if results[0].boxes.id is None:
            continue

        track_ids         = results[0].boxes.id.int().cpu().tolist()
        boxes_xyxy        = results[0].boxes.xyxy.cpu().tolist()
        boxes_cls         = results[0].boxes.cls.int().cpu().tolist()
        boxes_conf        = results[0].boxes.conf.cpu().tolist()
        current_track_ids = set(track_ids)

        for track_id, xyxy, yolo_class, conf in zip(track_ids, boxes_xyxy, boxes_cls, boxes_conf):
            if track_id in seen_track_ids:
                continue
            crop = crop_xyxy(frame, xyxy)
            if crop is None:
                continue
            _update_pending(pending_tracks, track_id, conf, crop, yolo_class)

        person_crops, person_track_ids, vehicle_crops, vehicle_track_ids = _flush_ready_tracks(
            pending_tracks, seen_track_ids, current_track_ids,
            save_crops, crops_dir, camera_id,
        )

        if person_crops:
            _register_persons(person_extractor, person_crops, person_track_ids,
                              camera_id, register_url, track_id_to_global_id, seen_track_ids)

        if vehicle_crops:
            _register_vehicles(vehicle_extractor, vehicle_crops, vehicle_track_ids,
                               camera_id, register_url, track_id_to_global_id, seen_track_ids)

    cap.release()
    print("[Client] Stopped.")


if __name__ == "__main__":
    main()
