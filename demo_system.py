"""
demo_system.py — composite demo video showing the full ReID system.

Layout (1280x720):
  Top    (360px): 3 camera feeds side by side, all running simultaneously
  Middle (200px): Live dashboard — metric cards + person/vehicle identity grids
  Footer (160px): PostgreSQL sightings table

Usage:
  python demo_system.py
  python demo_system.py --max-frames 500 --out demo_system.mp4
"""

import sys, argparse, base64, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
import cv2
import torch
import requests
import psycopg2
from pathlib import Path
from ultralytics import YOLO
from edge_client import build_vehicle_reid, build_person_reid, bgr_to_rgb_list, \
                        crop_xyxy, CLASS_PERSON, VEHICLE_CLASSES, REID_BUFFER_FRAMES

TRAIN_DIR    = Path("AIC21_Track2_ReID/image_train")
TRAIN_XML    = Path("AIC21_Track2_ReID/train_label.xml")
MARKET_TRAIN = Path("Market-1501/Market-1501-v15.09.15/bounding_box_train")

SERVER    = "http://127.0.0.1:5005"
DB_CONFIG = dict(host="127.0.0.1", port=5432, dbname="reid_logs",
                 user="postgres", password="707070")

SIM_VIDEOS = [
    (Path("sim_cam1.mp4"), "Camera-1"),
    (Path("sim_cam2.mp4"), "Camera-2"),
    (Path("sim_cam3.mp4"), "Camera-3"),
]
DEMO_THRESHOLD = 0.2

# ── Output frame layout ────────────────────────────────────────────────────────
OW, OH   = 1280, 720
CAM_H    = 360        # top strip: 3 camera feeds side by side
DASH_H   = 200        # middle strip: dashboard
FOOTER_H = 160        # bottom strip: SQL table  (360+200+160 = 720)
CAM_W    = OW // 3    # 426px per camera; last one gets OW - 2*CAM_W = 428
FPS_OUT  = 25

# ── Colors (BGR) ───────────────────────────────────────────────────────────────
C_BG     = (18, 18, 18)
C_PANEL  = (28, 28, 38)
C_CARD   = (42, 42, 55)
C_WHITE  = (240, 240, 240)
C_GREY   = (120, 120, 130)
C_ACCENT = (255, 180,  80)   # amber
C_GREEN  = ( 80, 200, 100)
C_TEAL   = ( 60, 220, 200)
C_PURPLE = (200,  80, 200)

PALETTE = [
    ( 50,200, 80),(80, 80,255),(  0,200,255),(255,200,  0),
    (180,  0,255),( 0,220,220),(255,140,  0),(255, 60,160),
]

def gid_color(gid: str):
    try:
        n = int(gid.split("-")[-1]) - 1
    except Exception:
        n = 0
    return PALETTE[n % len(PALETTE)]


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _shadow_text(img, text, xy, size, color, thick):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size,
                (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size,
                color, thick, cv2.LINE_AA)

def put_text(img, text, xy, size=0.5, color=C_WHITE, bold=False):
    _shadow_text(img, text, xy, size, color, 2 if bold else 1)

def rrect(img, x1, y1, x2, y2, color, r=6):
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(img, (cx,cy), r, color, -1)

def metric_card(panel, x, y, w, h, label, value, accent):
    rrect(panel, x, y, x+w, y+h, C_CARD)
    cv2.rectangle(panel, (x,y), (x+4,y+h), accent, -1)
    val_str = str(value)
    (tw, th), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    put_text(panel, val_str, (x+8, y+h//2+th//2), 0.85, C_WHITE, bold=True)
    put_text(panel, label,   (x+8, y+h-7),         0.37, C_GREY)


# ── API + DB helpers ───────────────────────────────────────────────────────────

def api_get(path, default=None):
    try:
        return requests.get(f"{SERVER}{path}", timeout=2).json()
    except Exception:
        return default

def fetch_crop_bgr(gid: str):
    try:
        r = requests.get(f"{SERVER}/identities/{gid}/crop", timeout=2)
        if r.status_code == 200:
            arr = np.frombuffer(r.content, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        pass
    return None

def fetch_sightings_from_db(n=5):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor()
        cur.execute("""
            SELECT timestamp, camera_id, global_id, category,
                   match_score, is_new_identity
            FROM sightings ORDER BY timestamp DESC LIMIT %s
        """, (n,))
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return []


# ── Dashboard (full 1280 × DASH_H) ────────────────────────────────────────────

def render_dashboard(panel: np.ndarray, stats: dict,
                     identities: list, crop_cache: dict, frame_no: int):
    panel[:] = C_PANEL

    # Title bar
    cv2.rectangle(panel, (0,0), (OW,20), (20,20,35), -1)
    put_text(panel, "ReID Central Server  —  Live Dashboard",
             (10,14), 0.5, C_ACCENT, bold=True)
    put_text(panel, f"frame {frame_no}", (OW-110,14), 0.42, C_GREY)

    # Metric cards
    ids   = stats.get("identities", {})
    cards = [
        ("PERSONS",   ids.get("person",  0), C_GREEN),
        ("VEHICLES",  ids.get("vehicle", 0), C_ACCENT),
        ("SIGHTINGS", stats.get("total_sightings", 0), C_TEAL),
        ("CAMERAS",   stats.get("active_cameras",  0), C_PURPLE),
    ]
    mw = (OW - 20) // 4 - 4
    mh = 44
    for i, (lbl, val, acc) in enumerate(cards):
        metric_card(panel, 8 + i*(mw+4), 22, mw, mh, lbl, val, acc)

    # Split identities into persons and vehicles
    persons  = [v for v in identities if v.get("global_id","").startswith("Person")][:13]
    vehicles = [v for v in identities if v.get("global_id","").startswith("Vehicle")][:13]

    gy0     = 22 + mh + 6        # ~72
    avail_h = DASH_H - gy0 - 4   # ~124

    def draw_row(items, y0, row_h, label_color):
        if not items:
            return
        n  = len(items)
        cw = max(60, (OW - 8) // n)
        n  = min(n, (OW - 8) // cw)
        ch = row_h - 2
        img_h = ch - 16
        for ci, v in enumerate(items[:n]):
            gid = v.get("global_id","?")
            cx  = 4 + ci * cw
            cy  = y0
            if cx + cw > OW:
                break
            rrect(panel, cx, cy, cx+cw-2, cy+ch, C_CARD)
            color = gid_color(gid)
            crop  = crop_cache.get(gid)
            if crop is not None:
                try:
                    # Preserve aspect ratio: fit crop into (target_w, img_h) and letterbox
                    target_w = cw - 4
                    target_h = img_h
                    h, w = crop.shape[:2]
                    if h == 0 or w == 0:
                        raise ValueError("Empty crop")
                    scale = min(target_w / w, target_h / h)
                    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                    resized = cv2.resize(crop, (nw, nh))
                    canvas_thumb = np.full((target_h, target_w, 3), (35, 35, 50), dtype=np.uint8)
                    xoff = (target_w - nw) // 2
                    yoff = (target_h - nh) // 2
                    canvas_thumb[yoff:yoff+nh, xoff:xoff+nw] = resized
                    panel[cy+2:cy+2+target_h, cx+2:cx+2+target_w] = canvas_thumb
                except Exception:
                    pass
            else:
                cv2.rectangle(panel,(cx+2,cy+2),(cx+cw-4,cy+img_h+2),(35,35,50),-1)
                put_text(panel,"...",(cx+cw//2-10,cy+img_h//2+5),0.45,C_GREY)
            cv2.rectangle(panel,(cx,cy+ch-14),(cx+cw-2,cy+ch),color,-1)
            put_text(panel, gid, (cx+3,cy+ch-3), 0.3, (255,255,255))
            sn = v.get("sightings",0)
            if sn:
                put_text(panel,f"{sn}x",(cx+cw-28,cy+ch-3),0.28,(220,220,220))

    if persons and vehicles:
        rh = avail_h // 2
        draw_row(persons,  gy0,      rh, C_GREEN)
        draw_row(vehicles, gy0 + rh, rh, C_ACCENT)
    elif persons:
        draw_row(persons,  gy0, avail_h, C_GREEN)
    elif vehicles:
        draw_row(vehicles, gy0, avail_h, C_ACCENT)
    else:
        put_text(panel, "Waiting for detections...", (12, gy0+24), 0.5, C_GREY)


# ── Footer (full 1280 × FOOTER_H) ─────────────────────────────────────────────

def render_footer(footer: np.ndarray, rows: list):
    footer[:] = (12, 12, 22)
    cv2.rectangle(footer, (0,0), (OW,26), (20,20,38), -1)
    put_text(footer,
             "PostgreSQL  --  SELECT timestamp, camera_id, global_id, category,"
             " match_score, is_new FROM sightings ORDER BY timestamp DESC LIMIT 5",
             (8,18), 0.42, (140,140,200))

    cols   = ["timestamp",  "camera_id", "global_id", "category", "score", "new"]
    widths = [195,           115,          110,          90,          70,     50]
    xs = [8]
    for w in widths[:-1]:
        xs.append(xs[-1]+w)

    hdr_y = 42
    cv2.line(footer, (0,28), (OW,28), (40,40,60), 1)
    for col, x in zip(cols, xs):
        put_text(footer, col, (x, hdr_y), 0.42, (160,160,210))
    cv2.line(footer, (0,hdr_y+6), (OW,hdr_y+6), (35,35,55), 1)

    for ri, row in enumerate(rows[:4]):
        ry = hdr_y + 22 + ri*24
        if ri % 2 == 0:
            cv2.rectangle(footer, (0,ry-16),(OW,ry+6),(18,18,32),-1)
        ts, cam, gid, cat, score, is_new = row
        vals   = [str(ts)[:19], str(cam)[:14], str(gid), str(cat),
                  f"{float(score):.3f}", "YES" if is_new else "no"]
        colors = [C_WHITE, C_WHITE, gid_color(gid), C_GREY,
                  C_WHITE, C_GREEN if is_new else C_GREY]
        for val, x, color in zip(vals, xs, colors):
            put_text(footer, val, (x, ry), 0.4, color)


# ── Vehicle gallery seeder ─────────────────────────────────────────────────────

def seed_gallery_from_train(v_ext, n: int = 12) -> int:
    if not TRAIN_XML.exists() or not TRAIN_DIR.exists():
        print("[Demo] Vehicle seed: AIC21 training data not found, skipping.")
        return 0

    raw = TRAIN_XML.read_bytes().replace(b'encoding="gb2312"', b'', 1)
    root_el = ET.fromstring(raw)
    vid_to_images: dict[str, list[str]] = {}
    for item in root_el.iter("Item"):
        vid = item.get("vehicleID")
        img = item.get("imageName")
        if vid and img:
            vid_to_images.setdefault(vid, []).append(img)

    all_vids = sorted(vid_to_images.keys())
    step     = max(1, len(all_vids) // n)
    selected = all_vids[::step][:n]

    try:
        requests.patch(f"{SERVER}/config", json={"vehicle": 1.1}, timeout=3)
    except Exception:
        pass

    print(f"[Demo] Seeding gallery with {len(selected)} training vehicles ...")
    seeded = 0
    for vid in selected:
        img_path = TRAIN_DIR / vid_to_images[vid][0]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        with torch.no_grad():
            feat = v_ext(bgr_to_rgb_list([img])).cpu().numpy()[0]
        try:
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            crop_b64 = base64.b64encode(jpg).decode() if ok else None
        except Exception:
            crop_b64 = None
        try:
            r = requests.post(f"{SERVER}/register_track",
                              json={"camera_id":"Seed","yolo_class":2,
                                    "features":feat.tolist(),"crop_b64":crop_b64},
                              timeout=5)
            if r.ok:
                print(f"  vehicleID {vid} -> {r.json().get('global_id')}")
                seeded += 1
        except Exception as e:
            print(f"  Seed failed for {vid}: {e}")

    try:
        requests.patch(f"{SERVER}/config", json={"vehicle": DEMO_THRESHOLD}, timeout=3)
    except Exception:
        pass

    print(f"[Demo] Vehicle gallery seeded: {seeded} vehicles.")
    return seeded


# ── Person gallery seeder (Market-1501) ────────────────────────────────────────

def seed_persons_from_market(p_ext, n: int = 10) -> int:
    if not MARKET_TRAIN.exists():
        print(f"[Demo] Person seed: {MARKET_TRAIN} not found, skipping.")
        return 0

    pid_to_files: dict[str, list[Path]] = {}
    for f in sorted(MARKET_TRAIN.glob("*.jpg")):
        pid = f.name[:4]
        if pid in ("0000", "-001"):
            continue
        pid_to_files.setdefault(pid, []).append(f)

    all_pids = sorted(pid_to_files.keys())
    step     = max(1, len(all_pids) // n)
    selected = all_pids[::step][:n]

    try:
        requests.patch(f"{SERVER}/config", json={"person": 1.1}, timeout=3)
    except Exception:
        pass

    print(f"[Demo] Seeding gallery with {len(selected)} Market-1501 persons ...")
    seeded = 0
    for pid in selected:
        img_path = pid_to_files[pid][0]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        with torch.no_grad():
            feat = p_ext(bgr_to_rgb_list([img])).cpu().numpy()[0]
        try:
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            crop_b64 = base64.b64encode(jpg).decode() if ok else None
        except Exception:
            crop_b64 = None
        try:
            r = requests.post(f"{SERVER}/register_track",
                              json={"camera_id":"Seed","yolo_class":0,
                                    "features":feat.tolist(),"crop_b64":crop_b64},
                              timeout=5)
            if r.ok:
                print(f"  person {pid} -> {r.json().get('global_id')}")
                seeded += 1
        except Exception as e:
            print(f"  Seed failed for person {pid}: {e}")

    try:
        requests.patch(f"{SERVER}/config", json={"person": 0.6}, timeout=3)
    except Exception:
        pass

    print(f"[Demo] Person gallery seeded: {seeded} persons.")
    return seeded


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=500,
                    help="Frames to read from each camera (default 500)")
    ap.add_argument("--out", default="demo_system.mp4")
    ap.add_argument("--weight-path", default="vehicle_reid_osnet_v4.pth")
    ap.add_argument("--no-reset", action="store_true",
                    help="Do not reset the gallery before the demo")
    ap.add_argument("--seed-from-train", type=int, default=12, metavar="N",
                    help="Pre-seed with N AIC21 vehicles (default 12, 0=skip)")
    ap.add_argument("--seed-persons", type=int, default=10, metavar="N",
                    help="Pre-seed with N Market-1501 persons (default 10, 0=skip)")
    args = ap.parse_args()

    # ── Server check
    try:
        stats = requests.get(f"{SERVER}/stats", timeout=3).json()
        print(f"[Demo] Server online  |  gallery={stats.get('gallery_size')}")
    except Exception:
        print(f"[Demo] ERROR: Cannot reach server at {SERVER}")
        print("       Start it with:  py -3.14 server.py")
        return

    if not args.no_reset:
        try:
            requests.post(f"{SERVER}/gallery/reset", timeout=3)
            print("[Demo] Gallery reset.")
        except Exception:
            pass

    try:
        requests.patch(f"{SERVER}/config", json={"vehicle": DEMO_THRESHOLD}, timeout=3)
        print(f"[Demo] Vehicle threshold -> {DEMO_THRESHOLD} (demo mode)")
    except Exception:
        pass

    available = [(p, c) for p, c in SIM_VIDEOS if p.exists()]
    if not available:
        print("[Demo] ERROR: No sim_cam*.mp4 found.")
        return
    print(f"[Demo] Cameras: {[c for _,c in available]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Demo] device={device}")
    # One detector per camera so ByteTrack state is never shared across feeds
    detectors = [YOLO("yolov8n.pt") for _ in range(len(SIM_VIDEOS))]
    wp    = args.weight_path
    v_ext = build_vehicle_reid(device=device,
                               weight_path=wp if Path(wp).exists() else None)
    p_ext = build_person_reid(device=device)
    print("[Demo] Models loaded.")

    if args.seed_from_train > 0:
        seed_gallery_from_train(v_ext, n=args.seed_from_train)
    if args.seed_persons > 0:
        seed_persons_from_market(p_ext, n=args.seed_persons)

    # ── Open all cameras simultaneously
    caps = []
    for sim_path, cam_id in available:
        cap = cv2.VideoCapture(str(sim_path))
        if not cap.isOpened():
            print(f"  Cannot open {sim_path}")
            cap = None
        caps.append((cap, cam_id, sim_path))

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS_OUT, (OW, OH))
    if not writer.isOpened():
        print(f"[Demo] ERROR: Cannot open writer for {args.out}")
        return

    # ── Per-camera tracking state
    states = [{"tid_to_gid": {}, "pending": {}, "seen_ids": set(), "new_flash": {}}
              for _ in caps]

    canvas            = np.full((OH, OW, 3), C_BG, dtype=np.uint8)
    crop_cache: dict  = {}
    stats_cache: dict = {}
    identities_cache  = []
    sightings_cache   = []
    last_api_update   = -9999

    n_persons  = 0
    n_vehicles = 0

    print(f"\n[Demo] Recording {args.max_frames} frames with "
          f"{len(caps)} simultaneous cameras...")

    for frame_no in range(1, args.max_frames + 1):

        # ── Read one frame from each camera
        frames = []
        for cap, cam_id, sim_path in caps:
            if cap is None:
                frames.append(None)
                continue
            ret, f = cap.read()
            frames.append(f if ret else None)

        if all(f is None for f in frames):
            break

        # ── Process each camera panel
        for cam_idx, (src_frame, (cap, cam_id, sim_path), state, detector) in \
                enumerate(zip(frames, caps, states, detectors)):

            # camera slot geometry
            x0     = cam_idx * CAM_W
            pw     = CAM_W if cam_idx < len(caps)-1 else (OW - cam_idx*CAM_W)
            canvas[:CAM_H, x0:x0+pw] = C_BG

            if src_frame is None:
                put_text(canvas, f"{cam_id}: end of video",
                         (x0+8, CAM_H//2), 0.5, C_GREY)
                continue

            src_h, src_w = src_frame.shape[:2]
            scale = min(pw / src_w, CAM_H / src_h)
            nw, nh = int(src_w*scale), int(src_h*scale)
            disp  = cv2.resize(src_frame, (nw, nh))
            px    = x0 + (pw - nw) // 2
            py    = (CAM_H - nh) // 2
            canvas[py:py+nh, px:px+nw] = disp

            # YOLO detect + track
            results = detector.track(src_frame, persist=True, verbose=False,
                                     classes=[0,1,2,3])

            if results[0].boxes.id is not None:
                tids   = results[0].boxes.id.int().cpu().tolist()
                bxyxys = results[0].boxes.xyxy.cpu().tolist()
                bclss  = results[0].boxes.cls.int().cpu().tolist()
                bconfs = results[0].boxes.conf.cpu().tolist()
                current = set(tids)

                tid_to_gid = state["tid_to_gid"]
                pending    = state["pending"]
                seen_ids   = state["seen_ids"]
                new_flash  = state["new_flash"]

                for tid, xyxy, cls, conf in zip(tids, bxyxys, bclss, bconfs):
                    if tid in seen_ids:
                        continue
                    crop = crop_xyxy(src_frame, xyxy)
                    if crop is None:
                        continue
                    if tid not in pending:
                        pending[tid] = {"conf": conf, "crop": crop, "cls": cls, "n": 1}
                    else:
                        pending[tid]["n"] += 1
                        if conf > pending[tid]["conf"]:
                            pending[tid].update({"conf": conf, "crop": crop})

                p_crops, p_tids = [], []
                v_items = []   # (tid, crop)
                for tid in list(pending):
                    d = pending[tid]
                    if d["n"] >= REID_BUFFER_FRAMES or tid not in current:
                        pending.pop(tid)
                        seen_ids.add(tid)
                        if d["cls"] == CLASS_PERSON:
                            p_crops.append(d["crop"]); p_tids.append(tid)
                        elif d["cls"] in VEHICLE_CLASSES:
                            v_items.append((tid, d["crop"]))

                def _encode(img):
                    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    return base64.b64encode(jpg).decode() if ok else None

                def _register(yolo_class, feat, crop_img, tid):
                    nonlocal n_persons, n_vehicles
                    try:
                        r = requests.post(f"{SERVER}/register_track",
                                          json={"camera_id": cam_id,
                                                "yolo_class": yolo_class,
                                                "features": feat.tolist(),
                                                "crop_b64": _encode(crop_img)},
                                          timeout=5)
                        if r.ok:
                            gid = r.json().get("global_id")
                            tid_to_gid[tid] = gid
                            new_flash[tid] = int(FPS_OUT * 1.5)
                            if yolo_class == CLASS_PERSON:
                                n_persons += 1
                            else:
                                n_vehicles += 1
                    except Exception:
                        pass

                if p_crops:
                    with torch.no_grad():
                        feats = p_ext(bgr_to_rgb_list(p_crops)).cpu().numpy()
                    for tid, feat, crop_img in zip(p_tids, feats, p_crops):
                        _register(CLASS_PERSON, feat, crop_img, tid)

                if v_items:
                    with torch.no_grad():
                        feats = v_ext(bgr_to_rgb_list([c for _,c in v_items])).cpu().numpy()
                    for (tid, crop_img), feat in zip(v_items, feats):
                        _register(2, feat, crop_img, tid)

                # Draw bounding boxes
                for tid, xyxy, cls in zip(tids, bxyxys, bclss):
                    gid = tid_to_gid.get(tid)
                    if not gid:
                        continue
                    x1,y1,x2,y2 = xyxy
                    x1 = int(x1*scale)+px; y1 = int(y1*scale)+py
                    x2 = int(x2*scale)+px; y2 = int(y2*scale)+py
                    # clip to panel
                    x1=max(x0,min(x1,x0+pw-1)); x2=max(x0,min(x2,x0+pw-1))
                    y1=max(0, min(y1,CAM_H-1)); y2=max(0, min(y2,CAM_H-1))
                    color = gid_color(gid)
                    thick = 3 if tid in new_flash else 2
                    cv2.rectangle(canvas, (x1,y1),(x2,y2), color, thick)
                    label = f"  {gid}  "
                    (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,2)
                    cv2.rectangle(canvas,(x1,y1-th-6),(x1+tw,y1),color,-1)
                    cv2.putText(canvas,label,(x1,y1-3),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
                    if tid in new_flash:
                        cv2.putText(canvas,"NEW",(x1,y2+16),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.48,color,2,cv2.LINE_AA)
                        new_flash[tid] -= 1
                        if new_flash[tid] <= 0:
                            new_flash.pop(tid)

            # Camera label + progress bar
            cv2.rectangle(canvas, (x0,0),(x0+len(cam_id)*10+16, 22),(0,0,0),-1)
            put_text(canvas, cam_id, (x0+6, 16), 0.5, C_ACCENT)
            frac  = frame_no / args.max_frames
            bar_w = int(pw * frac)
            cv2.rectangle(canvas,(x0,CAM_H-4),(x0+bar_w,CAM_H),
                          gid_color(f"x-{cam_idx+1}"),-1)

        # Dividers between camera panels
        for ci in range(1, len(caps)):
            cv2.line(canvas,(ci*CAM_W,0),(ci*CAM_W,CAM_H),(60,60,80),2)

        # ── API + DB poll (once per second of output)
        if frame_no - last_api_update >= FPS_OUT:
            last_api_update = frame_no
            s = api_get("/stats", {})
            if s:
                stats_cache = s
            vs = api_get("/identities?category=vehicle", {})
            ps = api_get("/identities?category=person",  {})
            identities_cache = (ps.get("identities",[]) +
                                vs.get("identities",[]))
            for v in identities_cache:
                gid = v["global_id"]
                if gid not in crop_cache:
                    img = fetch_crop_bgr(gid)
                    if img is not None:
                        crop_cache[gid] = img
            sightings_cache = fetch_sightings_from_db(5)

        # ── Render dashboard + footer
        cv2.line(canvas,(0,CAM_H),(OW,CAM_H),(60,60,80),2)
        render_dashboard(canvas[CAM_H:CAM_H+DASH_H, :],
                         stats_cache, identities_cache, crop_cache, frame_no)
        cv2.line(canvas,(0,CAM_H+DASH_H),(OW,CAM_H+DASH_H),(60,60,80),2)
        render_footer(canvas[CAM_H+DASH_H:, :], sightings_cache)

        writer.write(canvas)

        if frame_no % (FPS_OUT * 5) == 0:
            ids = stats_cache.get("identities", {})
            print(f"  frame {frame_no:>4}/{args.max_frames}  "
                  f"persons={ids.get('person',0)}  "
                  f"vehicles={ids.get('vehicle',0)}  "
                  f"sightings={stats_cache.get('total_sightings',0)}")

    for cap, _, _ in caps:
        if cap:
            cap.release()

    try:
        requests.patch(f"{SERVER}/config", json={"vehicle": 0.5}, timeout=3)
    except Exception:
        pass

    writer.release()
    sz = Path(args.out).stat().st_size / 1_000_000
    print(f"\n[Demo] Done. {args.out}  ({sz:.0f} MB)")
    print(f"       {frame_no} frames  |  "
          f"vehicles registered: {n_vehicles}  |  persons: {n_persons}")


if __name__ == "__main__":
    main()
