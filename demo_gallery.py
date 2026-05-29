"""
demo_gallery.py — Vehicle ReID retrieval demo for presentations.

Uses CityFlowV2 training images (already-cropped vehicle crops with known
ground-truth vehicle IDs) to show cross-camera retrieval:
    query vehicle (one camera) → top-5 gallery matches (other cameras)

Green border = correct match (same vehicle ID).
Blue  border = distractor (different vehicle).

Usage:
    python demo_gallery.py
    python demo_gallery.py --weight-path vehicle_reid_osnet_v4.pth
    python demo_gallery.py --n-vehicles 12 --hold-secs 4 --out demo_gallery.mp4
"""

import argparse
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from edge_client import build_vehicle_reid, bgr_to_rgb_list

# ── Layout constants ────────────────────────────────────────────────────────
W, H       = 1280, 720
FPS        = 25
TOP_K      = 5
N_SLOTS    = TOP_K + 1          # query + top-K
MARGIN     = 15
SLOT_W     = (W - 2 * MARGIN - (N_SLOTS - 1) * 10) // N_SLOTS   # ≈ 200
SLOT_GAP   = (W - 2 * MARGIN - N_SLOTS * SLOT_W) // (N_SLOTS - 1)

HEADER_H   = 90
IMG_H      = 380
FOOTER_H   = H - HEADER_H - IMG_H   # ≈ 250

# Colours (BGR)
BG          = (18, 18, 18)
HEADER_BG   = (30, 30, 30)
QUERY_CLR   = (50, 200, 100)      # green — query border
CORRECT_CLR = (40, 180, 60)       # green — correct match
WRONG_CLR   = (90, 90, 200)       # blue  — wrong match
SEP_CLR     = (60, 60, 60)
TEXT_CLR    = (230, 230, 230)
DIM_CLR     = (130, 130, 130)
WHITE       = (255, 255, 255)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX

DATASET_DIR = Path("AIC21_Track2_ReID")
IMG_DIR     = DATASET_DIR / "image_train"
LABEL_XML   = DATASET_DIR / "train_label.xml"

BATCH_SIZE  = 64
MIN_CAMS    = 2      # vehicle must appear in ≥ MIN_CAMS cameras to be a demo candidate
N_DISTRACTORS = 120  # distractor images per demo slide


# ── Helpers ─────────────────────────────────────────────────────────────────

def parse_labels(xml_path: Path):
    """Returns {imageName: (vehicleID, cameraID)}."""
    # GB2312 encoding; re-declare as utf-8 after reading bytes so ElementTree parses it
    raw = xml_path.read_bytes()
    raw = raw.replace(b'encoding="gb2312"', b'encoding="utf-8"')
    root = ET.fromstring(raw)
    result = {}
    for item in root.iter("Item"):
        result[item.attrib["imageName"]] = (
            item.attrib["vehicleID"],
            item.attrib["cameraID"],
        )
    return result


def slot_x(i: int) -> int:
    """Left x-coordinate of slot i (0-indexed)."""
    return MARGIN + i * (SLOT_W + SLOT_GAP)


def place_image(frame, img_bgr, x, y, w, h, border_color=None, border_thick=4):
    """Fit-and-pad img_bgr into an (w×h) cell at (x,y)."""
    if img_bgr is None or img_bgr.size == 0:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
        return
    ih, iw = img_bgr.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    ox = x + (w - nw) // 2
    oy = y + (h - nh) // 2
    # Dark background for the cell
    cv2.rectangle(frame, (x, y), (x + w, y + h), (28, 28, 28), -1)
    frame[oy:oy + nh, ox:ox + nw] = resized
    if border_color:
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, border_thick)


def put_text_centered(frame, text, cx, y, font=FONT, scale=0.55, color=TEXT_CLR, thick=1):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(frame, text, (cx - tw // 2, y + th), font, scale, color, thick, cv2.LINE_AA)


def render_slide(frame, query_img, gallery_imgs, gallery_metas, vid_id, slide_idx, total_slides):
    """
    query_img    : BGR ndarray
    gallery_imgs : list of BGR ndarrays (length TOP_K)
    gallery_metas: list of dicts with keys score, is_correct, cam_id
    """
    frame[:] = BG

    # ── Header ──────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (W, HEADER_H), HEADER_BG, -1)
    cv2.putText(frame, "Vehicle Re-Identification | Cross-Camera Retrieval",
                (MARGIN, 38), FONT_BOLD, 0.75, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"Query vehicle ID: {vid_id}    |    "
                f"Slide {slide_idx + 1}/{total_slides}    |    "
                f"Fine-tuned OSNet-AIN (CityFlowV2 + VehicleX)",
                (MARGIN, 68), FONT, 0.48, DIM_CLR, 1, cv2.LINE_AA)
    cv2.line(frame, (0, HEADER_H - 1), (W, HEADER_H - 1), SEP_CLR, 1)

    # ── Slot column labels ───────────────────────────────────────────────────
    label_y = HEADER_H + 14
    put_text_centered(frame, "QUERY", slot_x(0) + SLOT_W // 2, label_y,
                      scale=0.55, color=QUERY_CLR, thick=1)
    for k in range(TOP_K):
        put_text_centered(frame, f"Rank {k + 1}", slot_x(k + 1) + SLOT_W // 2, label_y,
                          scale=0.50, color=DIM_CLR)

    # ── Images ───────────────────────────────────────────────────────────────
    img_y = HEADER_H + 35
    place_image(frame, query_img, slot_x(0), img_y, SLOT_W, IMG_H,
                border_color=QUERY_CLR, border_thick=5)

    for k, (gimg, meta) in enumerate(zip(gallery_imgs, gallery_metas)):
        clr = CORRECT_CLR if meta["is_correct"] else WRONG_CLR
        place_image(frame, gimg, slot_x(k + 1), img_y, SLOT_W, IMG_H,
                    border_color=clr, border_thick=4)

    # ── Footer ───────────────────────────────────────────────────────────────
    footer_y = HEADER_H + 35 + IMG_H + 10

    # Similarity scores
    score_y = footer_y + 20
    put_text_centered(frame, "cosine sim", slot_x(0) + SLOT_W // 2, score_y,
                      scale=0.44, color=DIM_CLR)
    for k, meta in enumerate(gallery_metas):
        put_text_centered(frame, f"{meta['score']:.3f}",
                          slot_x(k + 1) + SLOT_W // 2, score_y,
                          scale=0.55, color=WHITE)

    # Camera IDs
    cam_y = score_y + 32
    for k, meta in enumerate(gallery_metas):
        put_text_centered(frame, meta["cam_id"],
                          slot_x(k + 1) + SLOT_W // 2, cam_y,
                          scale=0.44, color=DIM_CLR)

    # Match labels
    match_y = cam_y + 34
    for k, meta in enumerate(gallery_metas):
        if meta["is_correct"]:
            label, clr = "MATCH", CORRECT_CLR
        else:
            label, clr = "DISTRACTOR", WRONG_CLR
        put_text_centered(frame, label, slot_x(k + 1) + SLOT_W // 2, match_y,
                          scale=0.50, color=clr, thick=1)

    # Legend
    legend_y = H - 28
    cv2.line(frame, (0, H - 52), (W, H - 52), SEP_CLR, 1)
    for clr, label, x0 in [
        (QUERY_CLR,   "Query vehicle",    MARGIN),
        (CORRECT_CLR, "Correct match",    MARGIN + 180),
        (WRONG_CLR,   "Distractor",       MARGIN + 380),
    ]:
        cv2.rectangle(frame, (x0, legend_y - 10), (x0 + 18, legend_y + 6), clr, -1)
        cv2.putText(frame, label, (x0 + 26, legend_y + 4), FONT, 0.46, TEXT_CLR, 1, cv2.LINE_AA)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-path", default="vehicle_reid_osnet_v4.pth",
                    help="Fine-tuned OSNet-AIN checkpoint (.pth)")
    ap.add_argument("--n-vehicles", type=int, default=12)
    ap.add_argument("--hold-secs",  type=float, default=4.0,
                    help="Seconds to hold each retrieval slide")
    ap.add_argument("--out",        default="demo_gallery.mp4")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Demo] device={device}")

    # ── Parse labels ──────────────────────────────────────────────────────────
    print("[Demo] Parsing train_label.xml …")
    labels = parse_labels(LABEL_XML)                   # name → (vid, cam)
    by_vehicle = defaultdict(list)                     # vid → [name, ...]
    for name, (vid, cam) in labels.items():
        by_vehicle[vid].append(name)

    # Vehicles with ≥ MIN_CAMS distinct cameras
    multi_cam = []
    for vid, names in by_vehicle.items():
        cams = {labels[n][1] for n in names}
        if len(cams) >= MIN_CAMS:
            multi_cam.append((vid, names, cams))

    random.shuffle(multi_cam)
    demo_vehicles = multi_cam[:args.n_vehicles]
    print(f"[Demo] {len(multi_cam)} vehicles with ≥{MIN_CAMS} cameras; "
          f"selected {len(demo_vehicles)} for demo")

    # ── Collect images to embed ────────────────────────────────────────────────
    # For each demo vehicle: query image (1 cam) + gallery images (other cams)
    # + distractors from unrelated vehicles.
    demo_queries    = []   # list of (vehicle_idx, name)
    demo_galleries  = []   # list of [(name, is_correct), ...]

    all_distractor_names = []
    demo_vid_names = {n for _, names, _ in demo_vehicles for n in names}
    for name in labels:
        if name not in demo_vid_names:
            all_distractor_names.append(name)
    random.shuffle(all_distractor_names)

    for i, (vid, names, cams) in enumerate(demo_vehicles):
        by_cam = defaultdict(list)
        for n in names:
            by_cam[labels[n][1]].append(n)
        cam_list = list(by_cam.keys())
        query_cam = random.choice(cam_list)
        query_name = random.choice(by_cam[query_cam])

        # Gallery = all images from OTHER cameras of this vehicle + distractors
        positives = [n for n in names if labels[n][1] != query_cam]
        distractors = all_distractor_names[i * N_DISTRACTORS:(i + 1) * N_DISTRACTORS]
        gallery = [(n, True) for n in positives] + [(n, False) for n in distractors]
        random.shuffle(gallery)

        demo_queries.append((vid, query_name, query_cam))
        demo_galleries.append(gallery)

    # ── Embed all needed images ────────────────────────────────────────────────
    all_names_needed = set()
    for _, qname, _ in demo_queries:
        all_names_needed.add(qname)
    for gal in demo_galleries:
        for n, _ in gal:
            all_names_needed.add(n)

    all_names_needed = list(all_names_needed)
    print(f"[Demo] Loading & embedding {len(all_names_needed)} images …")

    extractor = build_vehicle_reid(device=device,
                                   weight_path=args.weight_path if Path(args.weight_path).exists() else None)

    emb_map = {}   # name → 512-dim np.ndarray
    for start in range(0, len(all_names_needed), BATCH_SIZE):
        batch_names = all_names_needed[start:start + BATCH_SIZE]
        batch_imgs  = []
        valid_names = []
        for n in batch_names:
            p = IMG_DIR / n
            img = cv2.imread(str(p))
            if img is not None:
                batch_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                valid_names.append(n)
        if not batch_imgs:
            continue
        with torch.no_grad():
            feats = extractor(batch_imgs).cpu().numpy()
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        for n, f in zip(valid_names, feats):
            emb_map[n] = f
        if (start // BATCH_SIZE) % 5 == 0:
            print(f"  {min(start + BATCH_SIZE, len(all_names_needed))}/{len(all_names_needed)}")

    print("[Demo] Embeddings done.")

    # ── Set up video writer ────────────────────────────────────────────────────
    writer = cv2.VideoWriter(args.out,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (W, H))
    hold_frames = int(args.hold_secs * FPS)
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    # ── Title card (2 sec) ────────────────────────────────────────────────────
    title_frame = np.full((H, W, 3), 18, dtype=np.uint8)
    cv2.putText(title_frame, "Vehicle Re-Identification",
                (W // 2 - 340, H // 2 - 60), FONT_BOLD, 1.4, WHITE, 2, cv2.LINE_AA)
    cv2.putText(title_frame, "Cross-Camera Retrieval Demo  |  CityFlowV2",
                (W // 2 - 270, H // 2 + 10), FONT_BOLD, 1.1, QUERY_CLR, 1, cv2.LINE_AA)
    cv2.putText(title_frame, "OSNet-AIN fine-tuned on CityFlowV2 + VehicleX",
                (W // 2 - 310, H // 2 + 70), FONT, 0.65, DIM_CLR, 1, cv2.LINE_AA)
    for _ in range(FPS * 2):
        writer.write(title_frame)

    # ── Per-vehicle slides ─────────────────────────────────────────────────────
    for slide_idx, ((vid, qname, qcam), gal) in enumerate(zip(demo_queries, demo_galleries)):
        if qname not in emb_map:
            continue
        q_feat = emb_map[qname]

        # Filter gallery to embedded images
        gal_valid = [(n, ok) for n, ok in gal if n in emb_map]
        if not gal_valid:
            continue

        G     = np.vstack([emb_map[n] for n, _ in gal_valid])
        sims  = (q_feat[None] @ G.T)[0]
        order = np.argsort(sims)[::-1][:TOP_K]

        top_names  = [gal_valid[i][0] for i in order]
        top_correct = [gal_valid[i][1] for i in order]
        top_scores  = [float(sims[i])  for i in order]
        top_cams    = [labels[n][1]    for n in top_names]

        query_img = cv2.imread(str(IMG_DIR / qname))
        gallery_imgs = [cv2.imread(str(IMG_DIR / n)) for n in top_names]
        gallery_metas = [
            {"score": s, "is_correct": ok, "cam_id": cam}
            for s, ok, cam in zip(top_scores, top_correct, top_cams)
        ]

        render_slide(frame, query_img, gallery_imgs, gallery_metas,
                     vid, slide_idx, len(demo_vehicles))

        for _ in range(hold_frames):
            writer.write(frame)

        n_correct = sum(m["is_correct"] for m in gallery_metas)
        print(f"  slide {slide_idx + 1:2d}  vehicle={vid}  "
              f"query_cam={qcam}  correct_in_top5={n_correct}/{TOP_K}")

    # ── End card ──────────────────────────────────────────────────────────────
    end_frame = np.full((H, W, 3), 18, dtype=np.uint8)
    cv2.putText(end_frame, "End of Demo",
                (W // 2 - 160, H // 2 - 20), FONT_BOLD, 1.2, WHITE, 1, cv2.LINE_AA)
    for _ in range(FPS * 2):
        writer.write(end_frame)

    writer.release()
    print(f"\n[Demo] Written → {args.out}")


if __name__ == "__main__":
    main()
