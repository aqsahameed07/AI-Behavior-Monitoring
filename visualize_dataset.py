"""
visualize_dataset.py

Creates preview grids of randomly sampled images from the phone detection
dataset with their YOLO bounding boxes drawn on top.

Output:
    reports/dataset_preview/train_preview.jpg
    reports/dataset_preview/val_preview.jpg
    reports/dataset_preview/test_preview.jpg
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_ROOT = Path("dataset/phone_detection")
OUTPUT_DIR = Path("reports/dataset_preview")

SPLITS = ["train", "val", "test"]

# Number of images to sample per split (3x3 grid = 9)
NUM_SAMPLES = 9
GRID_COLS = 3
GRID_ROWS = 3

# Size of each cell in the preview grid (width, height)
CELL_SIZE = (400, 400)

# Bounding box style
BOX_COLOR = (0, 255, 0)      # green
BOX_THICKNESS = 2
LABEL_COLOR = (0, 255, 0)
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.5
LABEL_THICKNESS = 1

# Class names (must match data.yaml)
CLASS_NAMES = {0: "phone"}

# Random seed for reproducible previews
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_yolo_labels(label_path: Path) -> list:
    """
    Read a YOLO label file.

    Returns a list of tuples: (class_id, x_center, y_center, width, height)
    All coordinates are normalized (0.0 - 1.0).
    """
    labels = []
    if not label_path.exists():
        return labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                labels.append((class_id, x, y, w, h))
            except ValueError:
                continue

    return labels


def draw_boxes(image, labels):
    """
    Draw YOLO bounding boxes on an image (in-place) and return it.
    """
    h_img, w_img = image.shape[:2]

    for (class_id, x_c, y_c, w, h) in labels:
        # Convert normalized center coords to pixel corners
        x1 = int((x_c - w / 2) * w_img)
        y1 = int((y_c - h / 2) * h_img)
        x2 = int((x_c + w / 2) * w_img)
        y2 = int((y_c + h / 2) * h_img)

        # Clamp to image bounds
        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y2 = max(0, min(y2, h_img - 1))

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # Draw label
        label_text = CLASS_NAMES.get(class_id, f"class_{class_id}")
        label_y = y1 - 5 if y1 - 5 > 10 else y1 + 15
        cv2.putText(
            image,
            label_text,
            (x1, label_y),
            LABEL_FONT,
            LABEL_SCALE,
            LABEL_COLOR,
            LABEL_THICKNESS,
            cv2.LINE_AA,
        )

    return image


def resize_with_padding(image, target_size):
    """
    Resize image to fit inside target_size while preserving aspect ratio,
    then pad with black bars to exactly match target_size.
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def build_preview_grid(image_paths, label_paths):
    """
    Build a GRID_ROWS x GRID_COLS grid of annotated images.
    """
    cell_w, cell_h = CELL_SIZE
    grid = np.zeros(
        (cell_h * GRID_ROWS, cell_w * GRID_COLS, 3),
        dtype=np.uint8,
    )

    for idx, (img_path, lbl_path) in enumerate(zip(image_paths, label_paths)):
        if idx >= GRID_ROWS * GRID_COLS:
            break

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [WARN] Could not read image: {img_path}")
            continue

        labels = load_yolo_labels(lbl_path)
        annotated = draw_boxes(image.copy(), labels)
        cell = resize_with_padding(annotated, CELL_SIZE)

        row = idx // GRID_COLS
        col = idx % GRID_COLS
        y0 = row * cell_h
        x0 = col * cell_w
        grid[y0:y0 + cell_h, x0:x0 + cell_w] = cell

        print(
            f"  [{idx + 1}/{GRID_ROWS * GRID_COLS}] "
            f"{img_path.name}  ->  {len(labels)} box(es)"
        )

    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DATASET VISUAL VERIFICATION")
    print("=" * 60)
    print(f"Dataset root: {DATASET_ROOT.resolve()}")
    print(f"Output dir  : {OUTPUT_DIR.resolve()}")
    print()

    if not DATASET_ROOT.exists():
        print(f"[ERROR] Dataset root not found: {DATASET_ROOT}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)

    for split in SPLITS:
        print("-" * 60)
        print(f"SPLIT: {split}")
        print("-" * 60)

        images_dir = DATASET_ROOT / "images" / split
        labels_dir = DATASET_ROOT / "labels" / split

        if not images_dir.exists():
            print(f"[SKIP] Image folder not found: {images_dir}")
            continue

        # Collect image files
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = sorted(
            [p for p in images_dir.iterdir() if p.suffix.lower() in image_exts]
        )

        if len(image_files) == 0:
            print(f"[SKIP] No images in {images_dir}")
            continue

        # Sample
        sample_count = min(NUM_SAMPLES, len(image_files))
        sampled = random.sample(image_files, sample_count)

        # Match labels
        label_files = []
        for img_path in sampled:
            lbl_path = labels_dir / (img_path.stem + ".txt")
            label_files.append(lbl_path)
            if not lbl_path.exists():
                print(f"  [WARN] Missing label for {img_path.name}")

        # Build preview grid
        grid = build_preview_grid(sampled, label_files)

        # Save
        out_path = OUTPUT_DIR / f"{split}_preview.jpg"
        cv2.imwrite(str(out_path), grid)
        print(f"\n  Saved: {out_path}\n")

    print("=" * 60)
    print("DONE. Inspect the previews in:")
    print(f"  {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()