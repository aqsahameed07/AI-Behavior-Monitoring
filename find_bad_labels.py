"""
find_bad_labels.py

Scans all YOLO label files and reports any annotation where:
    - width > 0.95  (box covers > 95% of image width)
    - height > 0.95 (box covers > 95% of image height)
    - width or height < 0.02 (box is tiny)
    - x_center or y_center outside [0.0, 1.0]
    - class_id is not 0

These are typically annotation errors that poison training.
"""

from pathlib import Path

DATASET = Path("dataset/phone_detection")
SPLITS = ["train", "val", "test"]

issues = []

for split in SPLITS:
    label_dir = DATASET / "labels" / split
    if not label_dir.exists():
        continue

    for label_file in label_dir.glob("*.txt"):
        try:
            with open(label_file) as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        issues.append(f"{label_file} line {line_num}: wrong field count ({len(parts)})")
                        continue

                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:])

                    if cls_id != 0:
                        issues.append(f"{label_file.name}:{line_num}  class_id={cls_id} (expected 0)")
                    if w > 0.95 or h > 0.95:
                        issues.append(f"{label_file.name}:{line_num}  oversized box w={w:.3f} h={h:.3f}")
                    if w < 0.02 or h < 0.02:
                        issues.append(f"{label_file.name}:{line_num}  tiny box w={w:.4f} h={h:.4f}")
                    if not (0.0 <= x_c <= 1.0) or not (0.0 <= y_c <= 1.0):
                        issues.append(f"{label_file.name}:{line_num}  center out of bounds ({x_c:.3f},{y_c:.3f})")
        except Exception as e:
            issues.append(f"{label_file}  ERROR: {e}")

if not issues:
    print("No issues found. Dataset labels are clean.")
else:
    print(f"Found {len(issues)} issues:\n")
    for issue in issues:
        print(f"  {issue}")
    print(f"\nTotal: {len(issues)} issues")
    print("\nAction: these label files should be quarantined before retraining.")