from pathlib import Path
import shutil

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset" / "phone_detection"
BACKUP_DIR = DATASET_DIR / "labels_backup_before_normalization"
SPLITS = ("train", "val", "test")


def main():
    if not DATASET_DIR.exists():
        print(f"ERROR: Dataset folder not found: {DATASET_DIR}")
        return

    if BACKUP_DIR.exists():
        print(f"STOP: Backup folder already exists: {BACKUP_DIR}")
        print("No files were changed. Keep the existing backup safe.")
        return

    label_dirs = [DATASET_DIR / "labels" / split for split in SPLITS]
    if not all(folder.is_dir() for folder in label_dirs):
        print("ERROR: One or more labels split folders are missing.")
        return

    label_files = []
    for folder in label_dirs:
        label_files.extend(folder.glob("*.txt"))

    if not label_files:
        print("ERROR: No label files found. Nothing was changed.")
        return

    # Validate every annotation before making any changes.
    parsed_files = []
    class_counts = {0: 0, 1: 0}

    for file_path in label_files:
        lines = file_path.read_text(encoding="utf-8").splitlines()
        new_lines = []

        for line_number, line in enumerate(lines, start=1):
            parts = line.strip().split()

            if len(parts) != 5:
                print(f"STOP: Invalid annotation in {file_path}, line {line_number}")
                return

            try:
                class_id = int(parts[0])
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                print(f"STOP: Non-numeric annotation in {file_path}, line {line_number}")
                return

            if class_id not in (0, 1):
                print(f"STOP: Unexpected class ID {class_id} in {file_path}, line {line_number}")
                return

            x_center, y_center, width, height = coords
            if (
                not all(0 <= value <= 1 for value in coords)
                or width <= 0
                or height <= 0
            ):
                print(f"STOP: Invalid bounding box in {file_path}, line {line_number}")
                return

            class_counts[class_id] += 1
            parts[0] = "0"  # The dataset has one class: phone.
            new_lines.append(" ".join(parts))

        parsed_files.append((file_path, new_lines))

    # Back up the complete labels directory before modifying any label.
    source_labels = DATASET_DIR / "labels"
    shutil.copytree(source_labels, BACKUP_DIR)

    try:
        for file_path, new_lines in parsed_files:
            file_path.write_text(
                "\n".join(new_lines) + ("\n" if new_lines else ""),
                encoding="utf-8"
            )
    except Exception:
        print("ERROR while writing labels. Restoring labels from backup...")
        shutil.rmtree(source_labels)
        shutil.copytree(BACKUP_DIR, source_labels)
        print("Original labels restored from backup.")
        return

    print("\n========== NORMALIZATION COMPLETE ==========")
    print(f"Label files processed: {len(parsed_files)}")
    print(f"Original class 0 annotations: {class_counts[0]}")
    print(f"Original class 1 annotations: {class_counts[1]}")
    print(f"Total annotations: {sum(class_counts.values())}")
    print("All annotations now use class ID 0 (phone).")
    print(f"Backup saved at: {BACKUP_DIR}")
    print("\nOriginal bounding-box coordinates were preserved.")
    print("Next: run python validate_dataset.py to verify all splits.")


if __name__ == "__main__":
    main()