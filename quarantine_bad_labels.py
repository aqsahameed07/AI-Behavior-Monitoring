"""
quarantine_bad_labels.py

Moves images and labels that have bad annotations to a quarantine folder.
Bad = oversized box (w>0.95 or h>0.95) or tiny box (w<0.02 or h<0.02).
"""

import shutil
from pathlib import Path

DATASET = Path("dataset/phone_detection")
QUARANTINE = Path("dataset/quarantine")
SPLITS = ["train", "val", "test"]

# Create quarantine folder
QUARANTINE.mkdir(parents=True, exist_ok=True)

moved_count = 0

for split in SPLITS:
    label_dir = DATASET / "labels" / split
    image_dir = DATASET / "images" / split
    if not label_dir.exists():
        continue

    for label_file in label_dir.glob("*.txt"):
        is_bad = False
        
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                w = float(parts[3])
                h = float(parts[4])
                if w > 0.95 or h > 0.95 or w < 0.02 or h < 0.02:
                    is_bad = True
                    break
        
        if is_bad:
            # Move label
            label_dest = QUARANTINE / f"{split}_labels_{label_file.name}"
            shutil.move(str(label_file), str(label_dest))
            
            # Move matching image
            for ext in [".jpg", ".jpeg", ".png"]:
                img_file = image_dir / f"{label_file.stem}{ext}"
                if img_file.exists():
                    img_dest = QUARANTINE / f"{split}_images_{img_file.name}"
                    shutil.move(str(img_file), str(img_dest))
                    break
            
            moved_count += 1

print(f"Moved {moved_count} bad image/label pairs to {QUARANTINE.resolve()}")