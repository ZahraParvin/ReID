import argparse
import os
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

# --- Main Application ---
def main():
    parser = argparse.ArgumentParser(description="ReID Edge Client")
    parser.add_argument("--camera-id", default="Camera-1")
    parser.add_argument("--video-source", default="test.mp4")
    parser.add_argument("--server-url", default="http://127.0.0.1:5005/register_track")
    parser.add_argument("--save-crops", action="store_true",
                        help="Save the best crop per track to disk for ground-truth annotation")
    parser.add_argument("--crops-dir", default="crops",
                        help="Directory to write crop images (default: crops)")
    parser.add_argument("--weight-path", default=None,
                        help="Path to fine-tuned OSNet vehicle checkpoint (.pth)")
    args = parser.parse_args()
    camera_id    = args.camera_id
    video_source = args.video_source
    server_url   = args.server_url
    save_crops   = args.save_crops
    crops_dir    = args.crops_dir
    weight_path  = args.weight_path

    if save_crops:
        os.makedirs(crops_dir, exist_ok=True)
        print(f"[Client] Saving crops to '{crops_dir}/'  (format: <camera>_track<id>_<timestamp>.jpg)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Detector (YOLOv8)
    detector = YOLO("yolov8n.pt")
    
    # 2. Load ReID Extractors (both use OSNet-AIN-x1.0, 512-dim, 256×256)
    person_extractor  = build_person_reid(device=device)
    vehicle_extractor = build_vehicle_reid(device=device, weight_path=weight_path)
    
    # 3. Video Capture
    cap = cv2.VideoCapture(video_source)
    window_name = f"Edge Client [{camera_id}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # 4. Tracking State
    seen_track_ids = set()
    track_id_to_global_id = {}
    # pending_tracks: track_id -> {best_conf, best_crop, yolo_class, frames_seen}
    pending_tracks = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 5. Run Detection & Tracking
        results = detector.track(frame, persist=True)
        annotated_frame = results[0].plot()  # Draw boxes

        # 6. Update best-crop buffer for unregistered tracks
        if results[0].boxes.id is None:
            cv2.imshow(window_name, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        track_ids  = results[0].boxes.id.int().cpu().tolist()
        boxes_xyxy = results[0].boxes.xyxy.cpu().tolist()
        boxes_cls  = results[0].boxes.cls.int().cpu().tolist()
        boxes_conf = results[0].boxes.conf.cpu().tolist()
        current_track_ids = set(track_ids)

        for track_id, xyxy, yolo_class, conf in zip(track_ids, boxes_xyxy, boxes_cls, boxes_conf):
            if track_id in seen_track_ids:
                continue
            crop = crop_xyxy(frame, xyxy)
            if crop is None:
                continue
            if track_id not in pending_tracks:
                pending_tracks[track_id] = {
                    "best_conf": conf, "best_crop": crop,
                    "yolo_class": yolo_class, "frames_seen": 1
                }
            else:
                pending_tracks[track_id]["frames_seen"] += 1
                if conf > pending_tracks[track_id]["best_conf"]:
                    pending_tracks[track_id]["best_conf"] = conf
                    pending_tracks[track_id]["best_crop"] = crop

        # 7. Extract Features & Register with Server (in batches)
        # Register tracks whose buffer is full OR that have disappeared this frame
        person_crops = []
        person_track_ids = []
        vehicle_crops = []
        vehicle_track_ids = []

        for track_id in list(pending_tracks):
            data = pending_tracks[track_id]
            if data["frames_seen"] >= REID_BUFFER_FRAMES or track_id not in current_track_ids:
                pending_tracks.pop(track_id)
                seen_track_ids.add(track_id)
                if save_crops:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"{camera_id}_track{track_id}_{ts}.jpg"
                    cv2.imwrite(os.path.join(crops_dir, fname), data["best_crop"])
                if data["yolo_class"] == CLASS_PERSON:
                    person_crops.append(data["best_crop"])
                    person_track_ids.append(track_id)
                elif data["yolo_class"] in VEHICLE_CLASSES:
                    vehicle_crops.append(data["best_crop"])
                    vehicle_track_ids.append((track_id, data["yolo_class"]))
        
        # --- Process Persons ---
        if person_crops:
            with torch.no_grad():
                features = person_extractor(bgr_to_rgb_list(person_crops))  # [N, 512]

            for track_id, feat_vector, crop_bgr in zip(person_track_ids, features, person_crops):
                try:
                    _, jpg = cv2.imencode('.jpg', crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    payload = {
                        "camera_id": camera_id,
                        "yolo_class": CLASS_PERSON,
                        "features": feat_vector.tolist(),
                        "crop_b64": base64.b64encode(jpg).decode()
                    }
                    response = requests.post(server_url, json=payload, timeout=5)
                    if response.status_code == 200:
                        global_id = response.json().get("global_id")
                        track_id_to_global_id[track_id] = global_id
                        print(f"[Client] Person Track {track_id} -> {global_id}")
                except Exception as e:
                    print(f"Error connecting to server: {e}")
                    seen_track_ids.remove(track_id)  # Allow retry

        # --- Process Vehicles ---
        if vehicle_crops:
            with torch.no_grad():
                features = vehicle_extractor(bgr_to_rgb_list(vehicle_crops))  # [N, 512]

            for (track_id, yolo_class), feat_vector, crop_bgr in zip(vehicle_track_ids, features, vehicle_crops):
                try:
                    _, jpg = cv2.imencode('.jpg', crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    payload = {
                        "camera_id": camera_id,
                        "yolo_class": yolo_class,
                        "features": feat_vector.tolist(),
                        "crop_b64": base64.b64encode(jpg).decode()
                    }
                    response = requests.post(server_url, json=payload, timeout=5)
                    if response.status_code == 200:
                        global_id = response.json().get("global_id")
                        track_id_to_global_id[track_id] = global_id
                        print(f"[Client] Vehicle Track {track_id} -> {global_id}")
                except Exception as e:
                    print(f"Error connecting to server: {e}")
                    seen_track_ids.remove(track_id)  # Allow retry
        
        # 8. Draw Global IDs on Frame
        for track_id, xyxy in zip(track_ids, boxes_xyxy):
            global_id = track_id_to_global_id.get(track_id)
            if global_id:
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                label = f"ID: {global_id}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 9. Display
        cv2.imshow(window_name, annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("[Client] Stopped.")

if __name__ == "__main__":
    main()
