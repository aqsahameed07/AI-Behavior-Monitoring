"""
capture_hard_negatives.py

Captures pure negative images (empty label files) to reduce background
false positives in the phone detector.

Usage:
  python capture_hard_negatives.py

Controls:
  SPACE  — save current frame as a negative
  Q      — quit
  D      — toggle debug: show what YOLO currently sees (to help you pick
           frames where the model is false-firing)

Output:
  dataset/phone_detection/images/train/neg_<timestamp>_<n>.jpg
  dataset/phone_detection/labels/train/neg_<timestamp>_<n>.txt   (empty file)

Nothing is written to val/test — negatives should go in train so they
influence learning, not evaluation. Your val set must remain an honest
measurement of accuracy on real (labeled) images.

DO NOT use this script for phone/wallet/mouse images. Those need labels.
This script is for background-only scenes.
"""

import cv2
import time
import sys
from pathlib import Path
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "phone_detector"
    / "best_yolo26_3class.pt"
)

TRAIN_IMAGES = BASE_DIR / "dataset" / "phone_detection" / "images" / "train"
TRAIN_LABELS = BASE_DIR / "dataset" / "phone_detection" / "labels" / "train"

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# For debug overlay — use the same thresholds as your live script
DEBUG_CONFIDENCE = 0.25
CLASS_NAMES_OF_INTEREST = {"phone", "wallet", "mouse"}

WINDOW_NAME = "Hard Negative Capture"


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def check_paths():
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found: {MODEL_PATH}")
    if not TRAIN_IMAGES.exists():
        sys.exit(f"Train images folder not found: {TRAIN_IMAGES}")
    if not TRAIN_LABELS.exists():
        sys.exit(f"Train labels folder not found: {TRAIN_LABELS}")


# ---------------------------------------------------------------------------
# Webcam warm-up (same proven DSHOW pattern)
# ---------------------------------------------------------------------------

def open_webcam(camera_index, target_frames=3, max_attempts=100):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    print("Warming up webcam...")
    real_count = 0
    attempt = 0
    while real_count < target_frames and attempt < max_attempts:
        ret, frame = cap.read()
        attempt += 1
        if not ret:
            time.sleep(0.05)
            continue
        if frame.mean() > 15:
            real_count += 1
        else:
            time.sleep(0.05)

    if real_count < target_frames:
        cap.release()
        sys.exit("Could not get real frames from webcam.")

    print(f"Webcam ready. Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    return cap


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_negative(frame, saved_count):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"neg_{timestamp}_{saved_count:03d}"
    img_path = TRAIN_IMAGES / f"{stem}.jpg"
    lbl_path = TRAIN_LABELS / f"{stem}.txt"

    cv2.imwrite(str(img_path), frame)
    # Write an empty label file — explicit "no objects here"
    lbl_path.write_text("")

    return img_path, lbl_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    check_paths()

    print(f"Loading model for debug overlay: {MODEL_PATH.name}")
    model = YOLO(str(MODEL_PATH))

    cap = open_webcam(CAMERA_INDEX)

    print()
    print("=" * 60)
    print("HARD NEGATIVE CAPTURE")
    print("=" * 60)
    print("Controls:")
    print("  SPACE — save current frame as a negative")
    print("  D     — toggle debug overlay (what YOLO currently sees)")
    print("  Q     — quit")
    print()
    print("Tip: press D so you can SEE when the model is false-firing.")
    print("     Capture those exact scenes — desk, wall, background, etc.")
    print("     Do NOT put a real phone in these captures unless you")
    print("     want the model to learn it's NOT a phone (that's fine).")
    print("=" * 60)

    saved_count = 0
    debug_on = True
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.mean() < 10:
                continue

            display = frame.copy()

            if debug_on:
                # Show what YOLO currently thinks — helps pick frames
                results = model(
                    frame,
                    conf=DEBUG_CONFIDENCE,
                    verbose=False,
                )[0]

                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_name = str(
                        model.names[int(box.cls[0])]
                    ).lower().strip()

                    if cls_name not in CLASS_NAMES_OF_INTEREST:
                        continue

                    if cls_name == "phone":
                        color = (255, 0, 0)
                    elif cls_name == "wallet":
                        color = (0, 200, 0)
                    else:
                        color = (0, 220, 220)

                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        display,
                        f"{cls_name} {conf:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                    )

                cv2.putText(
                    display,
                    f"DEBUG ON  ({len(results.boxes)} raw detections)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                )
            else:
                cv2.putText(
                    display,
                    "DEBUG OFF",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2,
                )

            cv2.putText(
                display,
                f"Saved negatives: {saved_count}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            cv2.putText(
                display,
                "SPACE=save  D=debug  Q=quit",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                debug_on = not debug_on
                print(f"Debug overlay {'ON' if debug_on else 'OFF'}")
            elif key == ord(' '):
                img_path, lbl_path = save_negative(frame, saved_count)
                saved_count += 1
                print(
                    f"Saved [{saved_count:03d}] "
                    f"{img_path.name} (+ empty label)"
                )

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print()
        print(f"Done. Saved {saved_count} negative images.")
        print(f"Images → {TRAIN_IMAGES}")
        print(f"Labels → {TRAIN_LABELS}")
        if saved_count == 0:
            print("WARNING: you saved 0 images. Nothing was added to training.")
        elif saved_count < 30:
            print(
                f"Note: {saved_count} negatives is fewer than the 50–100 target. "
                f"Consider adding more for better coverage."
            )


if __name__ == "__main__":
    main()