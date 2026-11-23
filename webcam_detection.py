"""
webcam_detection_v6.py — Live three-class detection (phone / wallet / mouse).

Key differences from v5:
  - Uses the THREE-CLASS model (best.pt from phone_wallet_mouse_v1)
  - Model distinguishes phone vs. wallet vs. mouse instead of forcing every
    rectangle into the single "phone" bucket
  - Aspect-ratio filter only applies to the phone class
  - Wallet and mouse are drawn and counted separately (no filtering)
  - On-screen debug panel shows raw vs. accepted counts per class

Do NOT overwrite webcam_detection_v5.py — that is the phone-only fallback.
"""

import cv2
import time
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = (
    "D:/AI-Behaviour-Monitor/runs/detect/"
    "phone_wallet_mouse_v1/weights/best.pt"
)

# Global YOLO confidence threshold. Lower this to see more weak detections;
# per-class acceptance thresholds below decide what actually counts.
CONFIDENCE_THRESHOLD = 0.25

# Per-class thresholds for ACCEPTING a detection as real
CLASS_CONFIDENCE = {
    "phone":  0.50,
    "wallet": 0.35,
    "mouse":  0.35,
}

# Phone aspect-ratio window (width / height)
# Portrait in hand:   ~0.40 - 0.65
# Landscape on desk:  ~1.55 - 2.40
# Anything outside [0.30, 2.60] is almost certainly not a phone.
PHONE_MIN_ASPECT = 0.30
PHONE_MAX_ASPECT = 2.60

# Reject tiny boxes that cannot be a real phone
PHONE_MIN_AREA = 2500  # pixels²

WINDOW_NAME = "AI Behaviour Monitor - 3-Class Live"
CAMERA_INDEX = 0

# BGR colors
CLASS_COLORS = {
    "phone":  (255, 0, 0),    # blue
    "wallet": (0, 200, 0),    # green
    "mouse":  (0, 220, 220),  # yellow-ish
}
REJECT_COLOR = (128, 128, 128)


# ---------------------------------------------------------------------------
# Phone shape check (applied ONLY to phone class)
# ---------------------------------------------------------------------------

def is_phone_shaped(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False
    area = w * h
    if area < PHONE_MIN_AREA:
        return False
    aspect = w / h
    return PHONE_MIN_ASPECT <= aspect <= PHONE_MAX_ASPECT


# ---------------------------------------------------------------------------
# Webcam warm-up (same proven DSHOW pattern)
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

    print("✅ Webcam ready")
    return cap


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    class_names = set(
        model.names.values() if isinstance(model.names, dict) else model.names
    )
    required = {"phone", "wallet", "mouse"}
    if not required.issubset({str(n).lower() for n in class_names}):
        raise ValueError(
            f"Model is not the three-class model. Got classes: {model.names}. "
            f"Expected: {required}"
        )

    print(f"Classes: {model.names}")
    print(f"Global YOLO confidence: {CONFIDENCE_THRESHOLD}")
    print(f"Per-class acceptance:   {CLASS_CONFIDENCE}")
    print(f"Phone aspect window:    {PHONE_MIN_ASPECT} – {PHONE_MAX_ASPECT}")

    cap = open_webcam_with_real_frames(CAMERA_INDEX)
    if cap is None:
        return

    print("Press 'q' to quit.")

    prev_time = time.time()
    fps = 0.0
    total_frames = 0

    # Session counters
    accepted = {"phone": 0, "wallet": 0, "mouse": 0}
    rejected_low_conf = {"phone": 0, "wallet": 0, "mouse": 0}
    rejected_shape = {"phone": 0}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.mean() < 10:
                continue

            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            annotated = frame.copy()

            # Per-frame counters (what's visible right now)
            frame_counts = {"phone": 0, "wallet": 0, "mouse": 0}
            frame_rejected_shape = 0
            frame_rejected_low_conf = 0

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_name = str(
                        model.names[int(box.cls[0])]
                    ).lower().strip()

                    if cls_name not in CLASS_CONFIDENCE:
                        continue

                    # --- Per-class confidence gate ---
                    if conf < CLASS_CONFIDENCE[cls_name]:
                        rejected_low_conf[cls_name] += 1
                        frame_rejected_low_conf += 1
                        cv2.rectangle(annotated, (x1, y1), (x2, y2),
                                      REJECT_COLOR, 1)
                        cv2.putText(
                            annotated,
                            f"{cls_name} low {conf:.2f}",
                            (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            REJECT_COLOR, 1,
                        )
                        continue

                    # --- Phone-only shape gate ---
                    if cls_name == "phone" and not is_phone_shaped(
                        x1, y1, x2, y2
                    ):
                        rejected_shape["phone"] += 1
                        frame_rejected_shape += 1
                        cv2.rectangle(annotated, (x1, y1), (x2, y2),
                                      REJECT_COLOR, 1)
                        cv2.putText(
                            annotated,
                            f"phone bad-shape {conf:.2f}",
                            (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            REJECT_COLOR, 1,
                        )
                        continue

                    # --- Accepted detection ---
                    accepted[cls_name] += 1
                    frame_counts[cls_name] += 1
                    color = CLASS_COLORS[cls_name]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"{cls_name.upper()} {conf:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    )

            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time
            total_frames += 1

            # --- Dashboard ---
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(
                annotated,
                f"PHONE  {frame_counts['phone']}  "
                f"(rej conf {frame_rejected_low_conf} / "
                f"rej shape {frame_rejected_shape})",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLASS_COLORS["phone"], 2,
            )
            cv2.putText(
                annotated,
                f"WALLET {frame_counts['wallet']}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLASS_COLORS["wallet"], 2,
            )
            cv2.putText(
                annotated,
                f"MOUSE  {frame_counts['mouse']}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLASS_COLORS["mouse"], 2,
            )

            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nSession summary:")
        print(f"  Frames processed:      {total_frames}")
        print(f"  Accepted phone:        {accepted['phone']}")
        print(f"  Accepted wallet:       {accepted['wallet']}")
        print(f"  Accepted mouse:        {accepted['mouse']}")
        print(f"  Phone rejected (conf): {rejected_low_conf['phone']}")
        print(f"  Phone rejected (shape):{rejected_shape['phone']}")
        print(f"  Wallet rejected (conf):{rejected_low_conf['wallet']}")
        print(f"  Mouse rejected (conf): {rejected_low_conf['mouse']}")


if __name__ == "__main__":
    main()