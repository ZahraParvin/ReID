"""
sim_cameras.py — Build 3 × 1-minute simulation videos for the ReID demo.

Sources
-------
* Vehicles : AIC21_Track2_ReID/image_test  (one crop per track)
* Persons  : Market-1501/.../bounding_box_test  (one crop per identity)

Design
------
* 3 output cameras, each exactly 60 s (1500 frames @ 25 FPS)
* 2 independent lanes per camera — up to 2 entities on-screen simultaneously
* Each entity visible for a random 0.5–1.5 s with no black gaps between them
* Each source image used exactly once across all three videos
* Each entity assigned to exactly one camera (never on two feeds at once)
* Lanes are spatially separated: Lane 0 on the left half, Lane 1 on the right half

Usage
-----
    python sim_cameras.py
    python sim_cameras.py --seed 7 --out-dir .
"""

import argparse, random, sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# ── Dataset paths ─────────────────────────────────────────────────────────────
VEHICLE_IMG_DIR  = Path("AIC21_Track2_ReID/image_test")
VEHICLE_TRACK_F  = Path("AIC21_Track2_ReID/test_track.txt")
PERSON_IMG_DIR   = Path("Market-1501/Market-1501-v15.09.15/bounding_box_test")

# ── Video parameters ──────────────────────────────────────────────────────────
FW, FH       = 1280, 720
FPS          = 25
DURATION_S   = 60
TOTAL_F      = FPS * DURATION_S    # 1500 frames
N_CAMS       = 3
N_LANES      = 2                   # simultaneous entities per camera
MIN_SHOW     = int(0.5 * FPS)      # 12 frames  (0.5 s)
MAX_SHOW     = int(1.5 * FPS)      # 37 frames  (1.5 s)

# ── Rendering parameters ──────────────────────────────────────────────────────
TARGET_H_V   = 240          # vehicle crop height in output frame
TARGET_H_P   = 210          # person crop height in output frame
BG           = (14, 14, 14)
BORDER_V     = (160, 160, 160)
BORDER_P     = ( 80, 200, 120)     # greenish tint for persons


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Entity:
    category: str        # "vehicle" or "person"
    path: Path

@dataclass
class Slot:
    start: int           # inclusive frame index
    end:   int           # exclusive frame index
    entity: Entity
    x: int               # top-left x in output frame
    y: int               # top-left y in output frame


# ── Entity loaders ────────────────────────────────────────────────────────────

def load_vehicle_entities() -> List[Entity]:
    """One entity per test track (middle frame)."""
    if not VEHICLE_TRACK_F.exists():
        print(f"[Sim] WARNING: {VEHICLE_TRACK_F} not found — no vehicle entities.")
        return []
    entities = []
    for line in VEHICLE_TRACK_F.read_text().splitlines():
        imgs = line.strip().split()
        if not imgs:
            continue
        rep  = imgs[len(imgs) // 2]
        path = VEHICLE_IMG_DIR / rep
        if path.exists():
            entities.append(Entity("vehicle", path))
    print(f"[Sim] Vehicles loaded: {len(entities)} tracks")
    return entities


def load_person_entities() -> List[Entity]:
    """One entity per unique person ID (first image of each ID)."""
    if not PERSON_IMG_DIR.exists():
        print(f"[Sim] WARNING: {PERSON_IMG_DIR} not found — no person entities.")
        return []
    pid_to_file: dict[str, Path] = {}
    for f in sorted(PERSON_IMG_DIR.glob("*.jpg")):
        pid = f.name[:4]
        if pid in ("0000", "-001"):   # junk / background
            continue
        if pid not in pid_to_file:
            pid_to_file[pid] = f
    entities = [Entity("person", p) for p in pid_to_file.values()]
    print(f"[Sim] Persons loaded: {len(entities)} identities")
    return entities


# ── Frame renderer ────────────────────────────────────────────────────────────

def _render_entity(frame: np.ndarray, entity: Entity,
                   x: int, y: int, img_cache: dict) -> None:
    """Paste a (cached) entity crop onto frame."""
    crop = img_cache.get(entity.path)
    if crop is None:
        return

    target_h = TARGET_H_V if entity.category == "vehicle" else TARGET_H_P
    ih, iw   = crop.shape[:2]
    scale    = target_h / max(ih, 1)
    nw, nh   = max(1, int(iw * scale)), target_h

    resized  = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

    x1 = max(0, min(x, FW - nw))
    y1 = max(0, min(y, FH - nh))
    x2, y2 = x1 + nw, y1 + nh

    frame[y1:y2, x1:x2] = resized
    border = BORDER_P if entity.category == "person" else BORDER_V
    cv2.rectangle(frame, (x1-1, y1-1), (x2, y2), border, 1)

    # small category label
    label = "P" if entity.category == "person" else "V"
    cv2.putText(frame, label, (x1+2, y1+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, border, 1, cv2.LINE_AA)


# ── Lane slot builder ─────────────────────────────────────────────────────────

def build_lane_slots(entities: List[Entity], lane_idx: int,
                     rng: random.Random) -> List[Slot]:
    """
    Fill [0, TOTAL_F) with back-to-back slots — no gaps.
    Lane 0 occupies the left half; Lane 1 the right half.
    """
    # horizontal zones (pixels)
    if lane_idx == 0:
        x_lo, x_hi = 30,       FW // 2 - 260
    else:
        x_lo, x_hi = FW // 2,  FW - 310

    x_lo = max(0, x_lo)
    x_hi = max(x_lo + 10, x_hi)

    slots: List[Slot] = []
    frame = 0
    for entity in entities:
        if frame >= TOTAL_F:
            break
        duration = rng.randint(MIN_SHOW, MAX_SHOW)
        end      = min(frame + duration, TOTAL_F)

        target_h = TARGET_H_V if entity.category == "vehicle" else TARGET_H_P
        x = rng.randint(x_lo, x_hi)
        y = rng.randint(FH // 5, FH - target_h - 20)

        slots.append(Slot(frame, end, entity, x, y))
        frame = end

    return slots


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".", help="Output directory")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--force",   action="store_true", help="Overwrite existing videos")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and shuffle all entities ─────────────────────────────────────────
    vehicles = load_vehicle_entities()
    persons  = load_person_entities()

    all_entities: List[Entity] = vehicles + persons
    rng.shuffle(all_entities)
    print(f"[Sim] Total unique entities: {len(all_entities)}")

    # ── How many entities does each lane need? ─────────────────────────────────
    # Worst case: all MAX_SHOW → ceil(TOTAL_F / MIN_SHOW) = ceil(1500/12) = 125
    # Expected (avg 24.5 frames): ceil(1500/24.5) ≈ 62
    # We use 100 per lane to be safe, so 6 lanes × 100 = 600 total.
    entities_per_lane = max(100, TOTAL_F // MIN_SHOW + 5)
    needed = N_CAMS * N_LANES * entities_per_lane

    if len(all_entities) < needed:
        # Allow repeats only if we truly run short (shouldn't happen with these datasets)
        factor = (needed // len(all_entities)) + 1
        all_entities = (all_entities * factor)[:needed]
        print(f"[Sim] WARNING: not enough unique images; some may repeat.")
    else:
        all_entities = all_entities[:needed]

    # ── Assign entities to camera × lane groups ───────────────────────────────
    # Layout: groups[cam][lane] = list of entities
    groups: List[List[List[Entity]]] = [
        [[] for _ in range(N_LANES)] for _ in range(N_CAMS)
    ]
    for i, entity in enumerate(all_entities):
        cam  = (i // N_LANES) % N_CAMS
        lane = i % N_LANES
        groups[cam][lane].append(entity)

    # ── Build slot schedules ───────────────────────────────────────────────────
    # slots[cam][lane] = list of Slot
    slots: List[List[List[Slot]]] = []
    for cam in range(N_CAMS):
        cam_slots = []
        for lane in range(N_LANES):
            lane_entities = groups[cam][lane]
            cam_slots.append(build_lane_slots(lane_entities, lane, rng))
        slots.append(cam_slots)

    # Print summary
    for cam in range(N_CAMS):
        total_ents = sum(len(slots[cam][l]) for l in range(N_LANES))
        n_veh = sum(1 for l in range(N_LANES)
                    for s in slots[cam][l] if s.entity.category == "vehicle")
        n_per = sum(1 for l in range(N_LANES)
                    for s in slots[cam][l] if s.entity.category == "person")
        print(f"  Camera-{cam+1}: {total_ents} appearances  "
              f"(vehicles={n_veh}, persons={n_per})")

    # ── Pre-load all images ────────────────────────────────────────────────────
    print("[Sim] Loading images ...")
    img_cache: dict[Path, Optional[np.ndarray]] = {}
    for cam in range(N_CAMS):
        for lane in range(N_LANES):
            for slot in slots[cam][lane]:
                if slot.entity.path not in img_cache:
                    img_cache[slot.entity.path] = cv2.imread(str(slot.entity.path))
    loaded = sum(1 for v in img_cache.values() if v is not None)
    print(f"[Sim] Images loaded: {loaded}/{len(img_cache)}")

    # ── Write videos ───────────────────────────────────────────────────────────
    bg_frame = np.full((FH, FW, 3), BG, dtype=np.uint8)

    for cam in range(N_CAMS):
        out_path = out_dir / f"sim_cam{cam+1}.mp4"

        if out_path.exists() and not args.force:
            try:
                cap = cv2.VideoCapture(str(out_path))
                nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if nf >= TOTAL_F - 5:
                    print(f"[Sim] Skipping {out_path.name} (already {nf} frames)")
                    continue
            except Exception:
                pass

        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 FPS, (FW, FH))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for {out_path}")

        # Build frame-level lookup: frame → list of active Slots
        active: dict[int, List[Slot]] = defaultdict(list)
        for lane in range(N_LANES):
            for slot in slots[cam][lane]:
                for f in range(slot.start, slot.end):
                    active[f].append(slot)

        n_ents = sum(len(slots[cam][l]) for l in range(N_LANES))
        print(f"[Sim] Writing {out_path.name}  "
              f"({TOTAL_F} frames, {n_ents} appearances) ...")

        for f in range(TOTAL_F):
            frame = bg_frame.copy()
            for slot in active.get(f, []):
                _render_entity(frame, slot.entity, slot.x, slot.y, img_cache)
            writer.write(frame)

            if (f + 1) % (FPS * 10) == 0:
                print(f"  {f+1:>5}/{TOTAL_F}  ({(f+1)/FPS:.0f}s)")

        writer.release()
        sz = out_path.stat().st_size / 1_000_000
        print(f"  >> {out_path.name}  {TOTAL_F/FPS:.0f}s  {sz:.0f} MB")

    print("\n[Sim] Done.")
    print(f"  Output: sim_cam1.mp4  sim_cam2.mp4  sim_cam3.mp4")
    print(f"  Each: {DURATION_S}s  ({TOTAL_F} frames @ {FPS} FPS)  1280x720")


if __name__ == "__main__":
    main()
