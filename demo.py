"""
demo.py — Self-contained ReID demo. No server or database required.

Runs YOLOv8 + ByteTrack + OSNet-AIN on a video, matches embeddings against
a local in-memory gallery (same match-or-create + EMA logic as the live
server), and writes an annotated video to disk.

Usage:
    python demo.py                             # test.mp4 -> demo_output.mp4
    python demo.py --video clip.mp4 --out out.mp4
    python demo.py --video 0                   # webcam
"""

import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from edge_client import (
    build_person_reid, build_vehicle_reid,
    bgr_to_rgb_list, crop_xyxy,
    CLASS_PERSON, VEHICLE_CLASSES, REID_BUFFER_FRAMES,
)

THRESHOLDS  = {"person": 0.6, "vehicle": 0.5}
EMA_ALPHA   = 0.9

# Colour palette — one colour per global ID (cycles after 20)
PALETTE = [
    (  0, 200,  50), (255,  80,  80), (  0, 140, 255), (255, 200,   0),
    (180,   0, 255), (  0, 220, 220), (255, 140,   0), ( 60, 180,  60),
    (255,  60, 160), ( 80, 200, 255), (200, 100,   0), (120, 255, 120),
    (255, 255,  80), (  0,  80, 200), (200,  50, 200), (255, 180, 100),
    (100, 255, 200), (200, 200,  50), (  0, 160, 120), (160,  80, 255),
]


def id_colour(global_id: str):
    n = int(global_id.split("-")[-1]) - 1
    return PALETTE[n % len(PALETTE)]


class LocalGallery:
    """Replicates the server's match-or-create + EMA logic in-process."""

    def __init__(self):
        self.entries  = {"person": [], "vehicle": []}
        self.counters = {"person": 0,  "vehicle": 0}

    def register(self, feat: np.ndarray, category: str):
        feat    = feat / (np.linalg.norm(feat) + 1e-8)
        gallery = self.entries[category]
        tau     = THRESHOLDS[category]

        if gallery:
            G    = np.vstack([e["feat"] for e in gallery])   # [M, D]
            sims = (feat[None] @ G.T)[0]                     # [M]
            best = int(np.argmax(sims))
            if float(sims[best]) >= tau:
                old = gallery[best]["feat"]
                upd = EMA_ALPHA * old + (1 - EMA_ALPHA) * feat
                gallery[best]["feat"] = upd / (np.linalg.norm(upd) + 1e-8)
                return gallery[best]["global_id"], False

        self.counters[category] += 1
        gid = f"{category.capitalize()}-{self.counters[category]}"
        gallery.append({"global_id": gid, "feat": feat})
        return gid, True


def draw_id(frame, xyxy, global_id, is_new):
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    colour = id_colour(global_id)
    label  = f"  {global_id}  "
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    # Label background
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), colour, -1)
    # Label text
    cv2.putText(frame, label, (x1, y1 - 5), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    # Flash "*NEW*" on first assignment
    if is_new:
        cv2.putText(frame, "*NEW*", (x1, y2 + 18), font, 0.55, colour, 2, cv2.LINE_AA)


def add_hud(frame, gallery, camera_id):
    persons  = gallery.counters["person"]
    vehicles = gallery.counters["vehicle"]
    lines = [
        f"Camera : {camera_id}",
        f"Persons : {persons}",
        f"Vehicles: {vehicles}",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (10, 30 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, txt, (10, 30 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",     default="test.mp4")
    parser.add_argument("--out",       default="demo_output.mp4")
    parser.add_argument("--camera-id", default="Camera-1")
    parser.add_argument("--weight-path", default=None,
                        help="Fine-tuned OSNet vehicle weights (.pth)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Demo] device={device}  source={args.video}  out={args.out}")

    detector          = YOLO("yolov8n.pt")
    person_extractor  = build_person_reid(device=device)
    vehicle_extractor = build_vehicle_reid(device=device, weight_path=args.weight_path)
    gallery           = LocalGallery()

    src = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open: {args.video}")

    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    print(f"[Demo] {W}x{H} @ {fps:.1f} fps  |  {total} frames")

    seen_ids       = set()
    tid_to_gid     = {}
    tid_to_new     = {}
    new_flash      = {}          # tid -> frames remaining to show *NEW*
    pending        = {}

    frame_n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1

        results = detector.track(frame, persist=True, verbose=False,
                                 classes=[0, 1, 2, 3])

        if results[0].boxes.id is None:
            add_hud(frame, gallery, args.camera_id)
            writer.write(frame)
            continue

        track_ids  = results[0].boxes.id.int().cpu().tolist()
        boxes_xyxy = results[0].boxes.xyxy.cpu().tolist()
        boxes_cls  = results[0].boxes.cls.int().cpu().tolist()
        boxes_conf = results[0].boxes.conf.cpu().tolist()
        current    = set(track_ids)

        # ── Buffer best crop ───────────────────────────────────────────────
        for tid, xyxy, cls, conf in zip(track_ids, boxes_xyxy, boxes_cls, boxes_conf):
            if tid in seen_ids:
                continue
            crop = crop_xyxy(frame, xyxy)
            if crop is None:
                continue
            if tid not in pending:
                pending[tid] = {"conf": conf, "crop": crop, "cls": cls, "n": 1}
            else:
                pending[tid]["n"] += 1
                if conf > pending[tid]["conf"]:
                    pending[tid]["conf"] = conf
                    pending[tid]["crop"] = crop

        # ── Flush buffers ──────────────────────────────────────────────────
        person_crops, person_tids   = [], []
        vehicle_crops, vehicle_tids = [], []
        for tid in list(pending):
            d = pending[tid]
            if d["n"] >= REID_BUFFER_FRAMES or tid not in current:
                pending.pop(tid)
                seen_ids.add(tid)
                if d["cls"] == CLASS_PERSON:
                    person_crops.append(d["crop"]); person_tids.append(tid)
                elif d["cls"] in VEHICLE_CLASSES:
                    vehicle_crops.append(d["crop"]); vehicle_tids.append((tid, d["cls"]))

        # ── Extract & match ────────────────────────────────────────────────
        if person_crops:
            with torch.no_grad():
                feats = person_extractor(bgr_to_rgb_list(person_crops)).cpu().numpy()
            for tid, feat in zip(person_tids, feats):
                gid, is_new = gallery.register(feat, "person")
                tid_to_gid[tid] = gid
                if is_new:
                    new_flash[tid] = int(fps * 1.5)   # show *NEW* for 1.5 s

        if vehicle_crops:
            with torch.no_grad():
                feats = vehicle_extractor(bgr_to_rgb_list(vehicle_crops)).cpu().numpy()
            for (tid, _), feat in zip(vehicle_tids, feats):
                gid, is_new = gallery.register(feat, "vehicle")
                tid_to_gid[tid] = gid
                if is_new:
                    new_flash[tid] = int(fps * 1.5)

        # ── Draw ───────────────────────────────────────────────────────────
        for tid, xyxy in zip(track_ids, boxes_xyxy):
            gid = tid_to_gid.get(tid)
            if not gid:
                continue
            is_new = tid in new_flash
            draw_id(frame, xyxy, gid, is_new)
            if tid in new_flash:
                new_flash[tid] -= 1
                if new_flash[tid] <= 0:
                    new_flash.pop(tid)

        add_hud(frame, gallery, args.camera_id)
        writer.write(frame)

        if frame_n % 100 == 0:
            p, v = gallery.counters["person"], gallery.counters["vehicle"]
            print(f"  frame {frame_n:>5}/{total}  persons={p}  vehicles={v}")

    cap.release()
    writer.release()
    p, v = gallery.counters["person"], gallery.counters["vehicle"]
    print(f"\n[Demo] Done. {args.out} — {p} persons, {v} vehicles identified.")


if __name__ == "__main__":
    main()
