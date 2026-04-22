"""
ecp_to_yolo.py
==============
Converts EuroCity Persons (ECP) annotations to YOLO format for fine-tuning
the YOLOv8 pedestrian detector used in edge_client.py.

ECP annotation format (one JSON per image):
  { "children": [ { "identity": "pedestrian", "x0": f, "y0": f, "x1": f, "y1": f, ... } ] }

Usage:
  python ecp_to_yolo.py --ecp-root /path/to/ECP --output-dir datasets/ecp_yolo
  python ecp_to_yolo.py --ecp-root /path/to/ECP --output-dir datasets/ecp_yolo --split day
  python ecp_to_yolo.py --ecp-root /path/to/ECP --output-dir datasets/ecp_yolo --min-height 20

After running, fine-tune with:
  yolo detect train data=datasets/ecp_yolo/ecp.yaml model=yolov8n.pt epochs=50 imgsz=640

ECP directory structure expected:
  <ecp-root>/
    day/   (and/or night/)
      img/
        train/<city>/<image>.png
        val/<city>/<image>.png
      labels/
        train/<city>/<image>.json
        val/<city>/<image>.json
"""

import argparse
import json
import os
import shutil
from pathlib import Path


# ECP identities to treat as "person" (YOLO class 0)
PERSON_IDENTITIES = {"pedestrian", "rider"}

# ECP identities to skip entirely (not useful for ReID or detection)
IGNORE_IDENTITIES = {"person-group-far-away", "bicycle-group", "buggy-group"}


def parse_args():
    p = argparse.ArgumentParser(description="Convert ECP dataset to YOLO format")
    p.add_argument("--ecp-root", required=True,
                   help="Root directory of the ECP dataset (contains day/ and/or night/)")
    p.add_argument("--output-dir", default="datasets/ecp_yolo",
                   help="Output directory for YOLO-format dataset (default: datasets/ecp_yolo)")
    p.add_argument("--split", choices=["day", "night", "both"], default="both",
                   help="Which lighting condition to include (default: both)")
    p.add_argument("--min-height", type=int, default=20,
                   help="Minimum bounding box height in pixels to include (default: 20)")
    p.add_argument("--min-width", type=int, default=8,
                   help="Minimum bounding box width in pixels to include (default: 8)")
    p.add_argument("--copy-images", action="store_true",
                   help="Copy images into output-dir (default: symlink/reference via YAML paths)")
    return p.parse_args()


def convert_label_file(json_path: Path, img_w: int, img_h: int, min_h: int, min_w: int):
    """
    Reads one ECP JSON annotation file and returns a list of YOLO label lines.
    Returns empty list if no valid annotations.
    """
    with open(json_path) as f:
        data = json.load(f)

    lines = []
    for child in data.get("children", []):
        identity = child.get("identity", "")
        if identity not in PERSON_IDENTITIES:
            continue

        x0 = float(child["x0"])
        y0 = float(child["y0"])
        x1 = float(child["x1"])
        y1 = float(child["y1"])

        # Clamp to image bounds
        x0 = max(0.0, min(x0, img_w))
        y0 = max(0.0, min(y0, img_h))
        x1 = max(0.0, min(x1, img_w))
        y1 = max(0.0, min(y1, img_h))

        box_w = x1 - x0
        box_h = y1 - y0

        if box_w < min_w or box_h < min_h:
            continue

        # YOLO format: class cx cy w h (normalised)
        cx = (x0 + x1) / 2.0 / img_w
        cy = (y0 + y1) / 2.0 / img_h
        nw = box_w / img_w
        nh = box_h / img_h

        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return lines


def process_split(ecp_root: Path, lighting: str, subset: str,
                  out_imgs: Path, out_labels: Path,
                  min_h: int, min_w: int, copy_images: bool) -> tuple[int, int]:
    """Process one (lighting, subset) combination. Returns (n_images, n_boxes)."""
    img_base   = ecp_root / lighting / "img"    / subset
    label_base = ecp_root / lighting / "labels" / subset

    if not img_base.exists() or not label_base.exists():
        print(f"  [skip] {lighting}/{subset} not found at {ecp_root}")
        return 0, 0

    n_images = 0
    n_boxes  = 0

    # Walk city subdirectories
    for city_dir in sorted(label_base.iterdir()):
        if not city_dir.is_dir():
            continue
        city = city_dir.name

        for json_path in sorted(city_dir.glob("*.json")):
            stem = json_path.stem  # filename without extension

            # Find the corresponding image (ECP uses .png)
            img_path = img_base / city / f"{stem}.png"
            if not img_path.exists():
                continue

            # We need image dimensions for normalisation; use a fast header read
            import struct
            import zlib
            try:
                img_w, img_h = _png_dimensions(img_path)
            except Exception:
                # Fall back: open with cv2 (slower but safe)
                import cv2
                tmp = cv2.imread(str(img_path))
                if tmp is None:
                    continue
                img_h, img_w = tmp.shape[:2]

            yolo_lines = convert_label_file(json_path, img_w, img_h, min_h, min_w)
            # Write label even if empty (YOLO trainer expects matching label files)
            out_label_path = out_labels / f"{lighting}_{city}_{stem}.txt"
            out_label_path.write_text("\n".join(yolo_lines))

            # Image: copy or record path
            out_img_path = out_imgs / f"{lighting}_{city}_{stem}.png"
            if copy_images:
                if not out_img_path.exists():
                    shutil.copy2(img_path, out_img_path)
            else:
                # Write a one-line pointer file that the YAML will reference by absolute path
                # Instead, we store the original absolute path in the YAML image list
                # For simplicity when not copying, we just symlink (Unix) or copy (Windows)
                if not out_img_path.exists():
                    try:
                        os.symlink(img_path.resolve(), out_img_path)
                    except (OSError, NotImplementedError):
                        shutil.copy2(img_path, out_img_path)

            n_images += 1
            n_boxes  += len(yolo_lines)

    return n_images, n_boxes


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Read PNG width/height from header bytes without loading the full image."""
    with open(path, "rb") as f:
        f.read(8)   # PNG signature
        f.read(4)   # chunk length
        f.read(4)   # chunk type (IHDR)
        w = int.from_bytes(f.read(4), "big")
        h = int.from_bytes(f.read(4), "big")
    return w, h


def write_yaml(output_dir: Path, lightings: list[str]):
    """Write the ecp.yaml file consumed by `yolo detect train data=...`."""
    yaml_path = output_dir / "ecp.yaml"
    content = f"""\
# EuroCity Persons → YOLO format
# Generated by ecp_to_yolo.py

path: {output_dir.resolve()}
train: images/train
val:   images/val

nc: 1
names:
  0: person
"""
    yaml_path.write_text(content)
    print(f"  Wrote {yaml_path}")
    return yaml_path


def main():
    args = parse_args()
    ecp_root   = Path(args.ecp_root)
    output_dir = Path(args.output_dir)

    lightings = ["day", "night"] if args.split == "both" else [args.split]

    for subset in ("train", "val"):
        (output_dir / "images"  / subset).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels"  / subset).mkdir(parents=True, exist_ok=True)

    total_imgs  = {"train": 0, "val": 0}
    total_boxes = {"train": 0, "val": 0}

    for lighting in lightings:
        for subset in ("train", "val"):
            print(f"Processing {lighting}/{subset} ...")
            n_imgs, n_boxes = process_split(
                ecp_root, lighting, subset,
                out_imgs   = output_dir / "images"  / subset,
                out_labels = output_dir / "labels"  / subset,
                min_h      = args.min_height,
                min_w      = args.min_width,
                copy_images= args.copy_images,
            )
            total_imgs[subset]  += n_imgs
            total_boxes[subset] += n_boxes
            print(f"  {n_imgs:,} images, {n_boxes:,} person boxes")

    yaml_path = write_yaml(output_dir, lightings)

    print("\nDone.")
    print(f"  Train: {total_imgs['train']:,} images, {total_boxes['train']:,} boxes")
    print(f"  Val:   {total_imgs['val']:,} images,  {total_boxes['val']:,} boxes")
    print(f"\nTo fine-tune YOLOv8n on ECP, run:")
    print(f"  yolo detect train data={yaml_path} model=yolov8n.pt epochs=50 imgsz=640 batch=16")
    print(f"  (The best weights will be saved to runs/detect/train/weights/best.pt)")
    print(f"  Then update edge_client.py: detector = YOLO('runs/detect/train/weights/best.pt')")


if __name__ == "__main__":
    main()
