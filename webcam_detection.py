"""
webcam_detection_v5.py — Live detection with aspect ratio filter + higher threshold.

Changes from v4:
  - CONFIDENCE_THRESHOLD raised to 0.70
  - Aspect ratio filter rejects square-ish boxes (wallets, hands, plates)
  - Real phones are elongated — wallets are square-ish
"""

import cv2
import time
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "D:/AI-Behaviour-Monitor/models/phone_detector/best_yolo26.pt"
CONFIDENCE_THRESHOLD = 0.70

# Aspect ratio filter (width / height)
# Phones in portrait: 0.4-0.6   | Phones in landscape: 1.6-2.4
# Wallets/hands/plates: 0.7-1.5
MIN_ASPECT = 0.40     # reject boxes wider than this ratio
MAX_ASPECT = 2.50     # reject boxes taller than this ratio
# Anything outside [0.40, 2.50] is not a phone

# Minimum box size (reject tiny detections)
MIN_BOX_AREA = 3000   # pixels² — filters tiny false positives

WINDOW_NAME = "AI Behaviour Monitor - Live Phone Detection"
CAMERA_INDEX = 0


# ---------------------------------------------------------------------------
# Helper — geometry filter
# ---------------------------------------------------------------------------

def is_phone_shaped(x1, y1, x2, y2):
    """Return True if the box aspect ratio is phone-like."""
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False
    aspect = w / h
    area = w * h
    if area < MIN_BOX_AREA:
        return False
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        return False
    return True


# ---------------------------------------------------------------------------
# Webcam warm-up (same as v4)
# ---------------------------------------------------------------------------

def open_webcam_with_real_frames(camera_index, target_frames=3, max_attempts=100):
    print(f"Opening webcam {camera_index} with DSHOW...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Warming up webcam until real frames arrive...")
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
            print(f"  Real frame #{real_count} (attempt {attempt})")
        else:
            if attempt % 20 == 0:
                print(f"  Still black at attempt {attempt}")
            time.sleep(0.05)
    if real_count < target_frames:
        print("ERROR: Could not get real frames.")
        cap.release()
        return None
    print(f"✅ Webcam ready")
    return cap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"Classes: {model.names}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Aspect ratio filter: {MIN_ASPECT} – {MAX_ASPECT}")

    cap = open_webcam_with_real_frames(CAMERA_INDEX)
    if cap is None:
        return

    print("Press 'q' to quit.")

    prev_time = time.time()
    fps = 0.0
    total_frames = 0
    total_detections = 0
    rejected_by_shape = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.mean() < 10:
                continue

            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            annotated = frame.copy()
            phone_count = 0
            max_conf = 0.0

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_name = model.names[int(box.cls[0])]

                    if cls_name != "phone":
                        continue

                    # --- Aspect ratio filter ---
                    if not is_phone_shaped(x1, y1, x2, y2):
                        rejected_by_shape += 1
                        # Draw in grey to show it was rejected (optional)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2),
                                      (128, 128, 128), 1)
                        cv2.putText(annotated, f"not phone {conf:.2f}",
                                    (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (128, 128, 128), 1)
                        continue

                    phone_count += 1
                    total_detections += 1
                    max_conf = max(max_conf, conf)
                    color = (0, 255, 0) if conf >= 0.80 else (0, 200, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f"phone {conf:.2f}",
                                (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time
            total_frames += 1

            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated, f"Phones: {phone_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated, f"Max conf: {max_conf:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated, f"Rejected: {rejected_by_shape}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession summary:")
        print(f"  Frames processed: {total_frames}")
        print(f"  Phone detections (passed): {total_detections}")
        print(f"  Rejected by shape: {rejected_by_shape}")


if __name__ == "__main__":
    main()