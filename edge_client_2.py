import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchreid
import requests  # For sending data to server
import time


# --- Configuration ---
SERVER_URL = "http://10.52.197.56:5005/register_track"
CAMERA_ID = "Camera-2"
VIDEO_SOURCE = "test.mp4"  # Or use 0 for webcam

# Map YOLO class indices to groups for ReID
CLASS_PERSON = 0
CLASS_BICYCLE = 1
CLASS_CAR = 2
CLASS_MOTORCYCLE = 3
VEHICLE_CLASSES = {CLASS_BICYCLE, CLASS_CAR, CLASS_MOTORCYCLE}
REID_BUFFER_FRAMES = 5  # collect this many frames per track, then register the highest-confidence crop

# --- ReID Models 
class VehicleReIDBackbone(torch.nn.Module):
    """
    ViT-B/16 backbone following TransReID (Shuting He et al., ICCV 2021).
    Extracts a 768-dim [CLS] token embedding, L2-normalised.
    Uses ImageNet-pretrained weights (torchvision). To use TransReID-finetuned
    weights, pass weight_path to the constructor.
    """
    def __init__(self, device="cpu", weight_path=None):
        super().__init__()
        self.device = device
        self.size = (224, 224)  # ViT-B/16 ImageNet native resolution
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Load ViT-B/16 with ImageNet weights
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.backbone = vit_b_16(weights=weights)
        self.backbone.heads = torch.nn.Identity()  # drop classification head; output is 768-dim CLS token

        if weight_path is not None:
            state = torch.load(weight_path, map_location="cpu")
            # TransReID checkpoints may be nested under 'model' key
            state = state.get("model", state)
            self.backbone.load_state_dict(state, strict=False)
            print(f"[VehicleReID] Loaded weights from {weight_path}")
        else:
            print("[VehicleReID] Using ImageNet pretrained ViT-B/16 (TransReID architecture).")

    def forward(self, imgs_bgr):
        if len(imgs_bgr) == 0:
            return torch.empty((0, 768), dtype=torch.float32, device=self.device)

        batch = []
        for img in imgs_bgr:
            rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, self.size, interpolation=cv2.INTER_LINEAR)
            x       = resized.astype(np.float32) / 255.0
            x       = (x - self.mean) / self.std
            x       = np.transpose(x, (2, 0, 1))  # HWC -> CHW
            batch.append(x)

        batch = torch.from_numpy(np.stack(batch, axis=0)).to(self.device)  # [N, 3, 224, 224]
        feats = self.backbone(batch)          # [N, 768]  (CLS token via Identity head)
        feats = F.normalize(feats, p=2, dim=1)
        return feats

def build_person_reid(device="cpu"):
    """Builds the OSNet-AIN model for Person ReID."""
    return torchreid.utils.FeatureExtractor(
        model_name="osnet_ain_x1_0",
        device=device,
        # pretrained=True is default, will auto-download weights
    )

def preprocess_person_imgs(imgs_bgr):
    """Converts BGR crops to RGB for torchreid."""
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Detector (YOLOv8)
    detector = YOLO("yolov8n.pt")
    
    # 2. Load ReID Extractors
    person_extractor = build_person_reid(device=device)
    vehicle_extractor = VehicleReIDBackbone(device=device).to(device).eval()
    
    # 3. Video Capture
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cv2.namedWindow("Edge Client - ReID System", cv2.WINDOW_NORMAL)
    
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
            cv2.imshow("Edge Client - ReID System", annotated_frame)
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
                if data["yolo_class"] == CLASS_PERSON:
                    person_crops.append(data["best_crop"])
                    person_track_ids.append(track_id)
                elif data["yolo_class"] in VEHICLE_CLASSES:
                    vehicle_crops.append(data["best_crop"])
                    vehicle_track_ids.append((track_id, data["yolo_class"]))
        
        # --- Process Persons ---
        if person_crops:
            person_rgbs = preprocess_person_imgs(person_crops)
            with torch.no_grad():
                features = person_extractor(person_rgbs)  # [N, 512]
            
            for track_id, feat_vector in zip(person_track_ids, features):
                try:
                    payload = {
                        "camera_id": CAMERA_ID,
                        "yolo_class": CLASS_PERSON,
                        "features": feat_vector.tolist()
                    }
                    
                    response = requests.post(SERVER_URL, json=payload, timeout=2)
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
                features = vehicle_extractor(vehicle_crops)  # [N, 768] (or placeholder dim)
                features = features.cpu().numpy()
            
            for (track_id, yolo_class), feat_vector in zip(vehicle_track_ids, features):
                try:
                    payload = {
                        "camera_id": CAMERA_ID,
                        "yolo_class": yolo_class,
                        "features": feat_vector.tolist()
                    }
                    
                    response = requests.post(SERVER_URL, json=payload, timeout=2)
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
        cv2.imshow("Edge Client - ReID System", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("[Client] Stopped.")

if __name__ == "__main__":
    main()
