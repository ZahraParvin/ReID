"""
evaluate.py — Offline cross-camera ReID evaluation.

Ground truth CSV (one row per crop):
    crop_path,true_identity,camera_id

Example rows:
    crops/cam1_alice.jpg,alice,Camera-1
    crops/cam2_alice.jpg,alice,Camera-2
    crops/cam1_bob.jpg,bob,Camera-1
    crops/cam2_bob.jpg,bob,Camera-2

Crops from --gallery-cam (default Camera-1) build the gallery;
all other camera crops are used as queries.

Usage:
    python evaluate.py --gt ground_truth.csv --category person
    python evaluate.py --gt ground_truth.csv --category person --sweep --output results.csv
    python evaluate.py --gt ground_truth.csv --category person --no-ema
    python evaluate.py --gt ground_truth.csv --category person --threshold 0.6
"""
import argparse
import csv
import sys

import cv2
import numpy as np
import torch

from edge_client import VehicleReIDBackbone, build_person_reid, preprocess_person_imgs

GALLERY_EMA_ALPHA = 0.9


def load_ground_truth(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def extract_features(rows, category, device):
    """Load all crops once and extract ReID features. Returns {crop_path: np.ndarray}."""
    seen = {}
    for row in rows:
        p = row["crop_path"]
        if p in seen:
            continue
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] Cannot read {p!r}, skipping.")
            continue
        seen[p] = img

    paths = list(seen.keys())
    images = [seen[p] for p in paths]
    if not images:
        return {}

    print(f"[Eval] Extracting features for {len(images)} crops ({category})...")
    if category == "person":
        extractor = build_person_reid(device=device)
        with torch.no_grad():
            feats = extractor(preprocess_person_imgs(images)).cpu().numpy()
    else:
        extractor = VehicleReIDBackbone(device=device).to(device).eval()
        with torch.no_grad():
            feats = extractor(images).cpu().numpy()

    return {p: feats[i] for i, p in enumerate(paths)}


def compute_ranking_metrics(q_feats, q_ids, g_feats, g_ids):
    """
    Compute Rank-1, Rank-5, mAP — threshold independent.

    For each query, ranks all gallery entries by cosine similarity and checks
    whether the correct identity appears in the top-k positions.
    Queries whose identity is absent from the gallery are skipped (standard CMC).
    """
    if not q_feats:
        return {"rank1": 0.0, "rank5": 0.0, "mAP": 0.0}

    Q_mat = np.stack(q_feats)   # [Q, D]
    G_mat = np.stack(g_feats)   # [G, D]
    sims  = Q_mat @ G_mat.T     # [Q, G]  (vectors are already L2-normalised)

    rank1 = rank5 = 0
    aps = []

    for q_idx, q_id in enumerate(q_ids):
        order   = np.argsort(-sims[q_idx])
        correct = np.array([g_ids[i] == q_id for i in order])

        if not correct.any():
            continue  # identity not in gallery — skip

        if correct[0]:
            rank1 += 1
        if correct[:5].any():
            rank5 += 1

        hits, prec_at_hits = 0, []
        for rank, is_hit in enumerate(correct, 1):
            if is_hit:
                hits += 1
                prec_at_hits.append(hits / rank)
        aps.append(np.mean(prec_at_hits))

    n = len(q_ids)
    return {
        "rank1": rank1 / n * 100,
        "rank5": rank5 / n * 100,
        "mAP":   np.mean(aps) * 100 if aps else 0.0,
    }


def compute_threshold_metrics(q_feats, q_ids, g_feats, g_ids, threshold, use_ema):
    """
    Simulate the live server's top-1 gallery matching at a fixed threshold.

    Per query:
      TP — matched to the correct identity  (sim >= threshold, same identity)
      FP — matched to the wrong identity    (sim >= threshold, different identity)
      FN — missed a match that should exist (sim < threshold, identity IS in gallery)
      TN — correctly withheld               (sim < threshold, identity NOT in gallery)
    """
    gallery = [{"features": f.reshape(1, -1).copy(), "true_identity": gid}
               for f, gid in zip(g_feats, g_ids)]
    gallery_id_set = set(g_ids)

    tp = fp = fn = tn = 0

    for feat, q_id in zip(q_feats, q_ids):
        feat_row     = feat.reshape(1, -1)
        should_match = q_id in gallery_id_set

        G_mat    = np.vstack([e["features"] for e in gallery])
        sims     = (feat_row @ G_mat.T)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= threshold:
            if gallery[best_idx]["true_identity"] == q_id:
                tp += 1
            else:
                fp += 1
            if use_ema:
                old     = gallery[best_idx]["features"]
                updated = GALLERY_EMA_ALPHA * old + (1 - GALLERY_EMA_ALPHA) * feat_row
                norm    = np.linalg.norm(updated, axis=1, keepdims=True)
                gallery[best_idx]["features"] = updated / np.maximum(norm, 1e-8)
        else:
            if should_match:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "threshold": threshold,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 1),
        "recall":    round(recall, 1),
        "f1":        round(f1, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-camera ReID offline evaluator")
    parser.add_argument("--gt",          required=True,
                        help="Ground truth CSV (crop_path,true_identity,camera_id)")
    parser.add_argument("--category",    choices=["person", "vehicle"], default="person")
    parser.add_argument("--gallery-cam", default="Camera-1",
                        help="Camera whose crops build the gallery (default: Camera-1)")
    parser.add_argument("--threshold",   type=float, default=None,
                        help="Evaluate at a single threshold; omit to sweep 0.20–0.90")
    parser.add_argument("--no-ema",      action="store_true",
                        help="Disable gallery EMA update during threshold simulation")
    parser.add_argument("--output",      default=None,
                        help="Save threshold sweep results to this CSV file")
    args = parser.parse_args()

    use_ema = not args.no_ema
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Eval] device={device}  category={args.category}  "
          f"ema={use_ema}  gallery-cam={args.gallery_cam}\n")

    rows     = load_ground_truth(args.gt)
    feat_map = extract_features(rows, args.category, device)

    gallery_entries = [(feat_map[r["crop_path"]], r["true_identity"])
                       for r in rows
                       if r["camera_id"] == args.gallery_cam
                       and r["crop_path"] in feat_map]
    query_entries   = [(feat_map[r["crop_path"]], r["true_identity"])
                       for r in rows
                       if r["camera_id"] != args.gallery_cam
                       and r["crop_path"] in feat_map]

    if not gallery_entries:
        print(f"[Eval] ERROR: No gallery crops found for camera '{args.gallery_cam}'.")
        sys.exit(1)
    if not query_entries:
        print("[Eval] ERROR: No query crops found (crops from cameras other than gallery-cam).")
        sys.exit(1)

    g_feats, g_ids = map(list, zip(*gallery_entries))
    q_feats, q_ids = map(list, zip(*query_entries))

    print(f"[Eval] Gallery : {len(g_feats)} crops | {len(set(g_ids))} identities")
    print(f"[Eval] Queries : {len(q_feats)} crops | {len(set(q_ids))} identities\n")

    # --- Ranking metrics (threshold-independent) ---
    r = compute_ranking_metrics(q_feats, q_ids, g_feats, g_ids)
    print("=== Ranking Metrics (threshold-free) ===")
    print(f"  Rank-1 : {r['rank1']:.1f}%")
    print(f"  Rank-5 : {r['rank5']:.1f}%")
    print(f"  mAP    : {r['mAP']:.1f}%\n")

    # --- Threshold-based metrics ---
    thresholds = (
        [round(t, 2) for t in np.arange(0.20, 0.91, 0.05)]
        if args.threshold is None
        else [args.threshold]
    )

    title = "=== Threshold Sweep ===" if len(thresholds) > 1 else f"=== Threshold {thresholds[0]:.2f} ==="
    header = f"{'Thresh':>7}  {'Prec%':>7}  {'Rec%':>7}  {'F1%':>7}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'TN':>5}"
    print(title)
    print(header)
    print("-" * len(header))

    results = []
    for t in thresholds:
        m = compute_threshold_metrics(q_feats, q_ids, g_feats, g_ids, t, use_ema)
        results.append(m)
        print(f"{m['threshold']:>7.2f}  {m['precision']:>6.1f}%  {m['recall']:>6.1f}%  "
              f"{m['f1']:>6.1f}%  {m['TP']:>5}  {m['FP']:>5}  {m['FN']:>5}  {m['TN']:>5}")

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[Eval] Results saved to {args.output}")


if __name__ == "__main__":
    main()
