"""
ecp_extract_crops.py
====================
Extracts pedestrian crops from the EuroCity Persons (ECP) dataset.
Crops can be used to:
  - Stress-test OSNet-AIN embeddings on diverse urban pedestrian appearances
  - Build a supplementary evaluation set (requires manual identity annotation)
  - Visualise embedding space coverage (e.g. t-SNE)

Usage:
  # Extract crops from ECP day/train, minimum 64px height:
  python ecp_extract_crops.py --ecp-root /path/to/ECP --output-dir ecp_crops

  # Also save embeddings as a .npz file for t-SNE / analysis:
  python ecp_extract_crops.py --ecp-root /path/to/ECP --output-dir ecp_crops --embed

  # Restrict to specific cities and cap at N crops per city:
  python ecp_extract_crops.py --ecp-root /path/to/ECP --output-dir ecp_crops \\
      --cities berlin amsterdam --max-per-city 200

Output:
  ecp_crops/
    <city>_<image>_<idx>.jpg     # one file per cropped pedestrian

  ecp_crops/embeddings.npz       # (if --embed) keys: paths, embeddings [N, 512]

Ground-truth annotation workflow (for evaluate.py):
  After extraction, open the crops folder and create ground_truth.csv:
    crop_path,true_identity,camera_id
    ecp_crops/berlin_img1_0.jpg,person_A,ecp_left
    ecp_crops/berlin_img1_1.jpg,person_B,ecp_right
  Then run:
    python evaluate.py --gt ground_truth.csv --category person
"""

import argparse
import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchreid


PERSON_IDENTITIES = {"pedestrian", "rider"}


def parse_args():
    p = argparse.ArgumentParser(description="Extract pedestrian crops from ECP")
    p.add_argument("--ecp-root", required=True,
                   help="Root of ECP dataset (contains day/ and/or night/)")
    p.add_argument("--output-dir", default="ecp_crops",
                   help="Where to save crop images (default: ecp_crops)")
    p.add_argument("--split", choices=["day", "night", "both"], default="day",
                   help="Lighting split to use (default: day)")
    p.add_argument("--subset", choices=["train", "val", "both"], default="val",
                   help="Dataset subset (default: val — smaller, quicker)")
    p.add_argument("--min-height", type=int, default=64,
                   help="Min bounding box height in pixels (default: 64)")
    p.add_argument("--min-width", type=int, default=24,
                   help="Min bounding box width in pixels (default: 24)")
    p.add_argument("--max-per-city", type=int, default=0,
                   help="Cap crops per city (0 = unlimited, default: 0)")
    p.add_argument("--cities", nargs="*", default=[],
                   help="Restrict to these city names (default: all cities)")
    p.add_argument("--embed", action="store_true",
                   help="Also extract OSNet-AIN embeddings and save as embeddings.npz")
    p.add_argument("--embed-batch", type=int, default=32,
                   help="Batch size for embedding extraction (default: 32)")
    return p.parse_args()


def load_person_reid(device):
    print("[ReID] Loading OSNet-AIN-x1.0 ...")
    extractor = torchreid.utils.FeatureExtractor(
        model_name="osnet_ain_x1_0",
        device=device,
    )
    print("[ReID] Model ready.")
    return extractor


def extract_crops_from_image(img_path: Path, json_path: Path,
                              min_h: int, min_w: int):
    """Returns list of (crop_bgr, x0, y0, x1, y1) for valid pedestrian boxes."""
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    img_h, img_w = img.shape[:2]

    with open(json_path) as f:
        data = json.load(f)

    crops = []
    for child in data.get("children", []):
        if child.get("identity", "") not in PERSON_IDENTITIES:
            continue
        x0 = max(0, int(float(child["x0"])))
        y0 = max(0, int(float(child["y0"])))
        x1 = min(img_w, int(float(child["x1"])))
        y1 = min(img_h, int(float(child["y1"])))

        if (x1 - x0) < min_w or (y1 - y0) < min_h:
            continue
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crops.append(crop)

    return crops


def embed_crops(extractor, crop_paths: list[str], batch_size: int) -> np.ndarray:
    """Run OSNet-AIN on saved crops and return [N, 512] float32 embeddings."""
    all_feats = []
    for i in range(0, len(crop_paths), batch_size):
        batch_paths = crop_paths[i:i + batch_size]
        batch_imgs = []
        for p in batch_paths:
            img = cv2.imread(p)
            if img is not None:
                batch_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not batch_imgs:
            continue
        with torch.no_grad():
            feats = extractor(batch_imgs)  # [B, 512]
        all_feats.append(feats.cpu().numpy())
        print(f"  Embedded {min(i + batch_size, len(crop_paths))}/{len(crop_paths)} crops", end="\r")

    print()
    return np.concatenate(all_feats, axis=0) if all_feats else np.empty((0, 512), dtype=np.float32)


def main():
    args = parse_args()
    ecp_root   = Path(args.ecp_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lightings = ["day", "night"] if args.split == "both" else [args.split]
    subsets   = ["train", "val"] if args.subset == "both" else [args.subset]
    city_filter = set(args.cities) if args.cities else None

    saved_paths = []
    total_crops = 0

    for lighting in lightings:
        for subset in subsets:
            label_base = ecp_root / lighting / "labels" / subset
            img_base   = ecp_root / lighting / "img"    / subset

            if not label_base.exists():
                print(f"[skip] {lighting}/{subset} not found")
                continue

            print(f"\nProcessing {lighting}/{subset} ...")

            for city_dir in sorted(label_base.iterdir()):
                if not city_dir.is_dir():
                    continue
                city = city_dir.name

                if city_filter and city not in city_filter:
                    continue

                city_count = 0
                for json_path in sorted(city_dir.glob("*.json")):
                    if args.max_per_city and city_count >= args.max_per_city:
                        break

                    stem     = json_path.stem
                    img_path = img_base / city / f"{stem}.png"
                    if not img_path.exists():
                        continue

                    crops = extract_crops_from_image(
                        img_path, json_path, args.min_height, args.min_width
                    )

                    for idx, crop in enumerate(crops):
                        if args.max_per_city and city_count >= args.max_per_city:
                            break
                        fname = f"{lighting}_{city}_{stem}_{idx:02d}.jpg"
                        out_path = output_dir / fname
                        cv2.imwrite(str(out_path), crop)
                        saved_paths.append(str(out_path))
                        city_count += 1

                total_crops += city_count
                print(f"  {city}: {city_count} crops")

    print(f"\nTotal crops saved: {total_crops} → {output_dir}/")

    if args.embed and saved_paths:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor = load_person_reid(device)
        print(f"\nExtracting embeddings (batch={args.embed_batch}, device={device}) ...")
        embeddings = embed_crops(extractor, saved_paths, args.embed_batch)

        npz_path = output_dir / "embeddings.npz"
        np.savez(npz_path, paths=np.array(saved_paths), embeddings=embeddings)
        print(f"Saved embeddings: {npz_path}  shape={embeddings.shape}")
        print("\nTo visualise with t-SNE:")
        print("  python -c \"")
        print("  import numpy as np; from sklearn.manifold import TSNE; import matplotlib.pyplot as plt")
        print(f"  d = np.load('{npz_path}', allow_pickle=True)")
        print("  z = TSNE(n_components=2, perplexity=30).fit_transform(d['embeddings'])")
        print("  plt.scatter(z[:,0], z[:,1], s=2); plt.title('OSNet-AIN on ECP'); plt.show()\"")

    print("\nNext steps:")
    print("  1. Annotate crops with identity labels → ground_truth.csv")
    print("  2. Run: python evaluate.py --gt ground_truth.csv --category person")
    print("  3. To fine-tune the detector:  python ecp_to_yolo.py --ecp-root", args.ecp_root)


if __name__ == "__main__":
    main()
