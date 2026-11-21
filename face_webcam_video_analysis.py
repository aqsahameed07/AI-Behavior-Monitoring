import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from face_detection import (
    MODEL_PATH,
    RIGHT_EYE,
    LEFT_EYE,
    RIGHT_IRIS,
    LEFT_IRIS,
    landmark_to_pixel,
    calculate_ear,
    calculate_iris_position,
    estimate_head_pose,
    draw_eye,
)


# ==========================================
# CONFIGURATION
# ==========================================

# --- Calibration ---
CALIBRATION_SECONDS = 3.0          # how long the user holds eyes open at startup
EAR_CLOSED_FRACTION = 0.30         # threshold = closed + 30% of (open - closed)

# --- Blink detection (uses RAW EAR, not smoothed) ---
MIN_BLINK_DURATION = 0.05          # real blinks can be 50ms at 60fps
MAX_BLINK_DURATION = 0.60
PROLONGED_CLOSURE_DURATION = 0.60
MIN_CLOSED_FRAMES = 2              # debounce: must be closed ≥2 frames

# --- Smoothing (display only) ---
SMOOTH_ALPHA = 0.35                # 0.35 * new + 0.65 * old

# --- Gaze ---
GAZE_LEFT_LIMIT = 0.35
GAZE_RIGHT_LIMIT = 0.65

# --- Head pose ---
HEAD_YAW_THRESHOLD = 20.0
HEAD_PITCH_THRESHOLD = 20.0

# --- Focus scoring weights (must sum to 1.0) ---
FOCUS_WEIGHT_GAZE = 0.40
FOCUS_WEIGHT_HEAD = 0.25
FOCUS_WEIGHT_EYES_OPEN = 0.20
FOCUS_WEIGHT_BLINK_RATE = 0.15

# Focus score smoothing window
FOCUS_SMOOTH_ALPHA = 0.10          # display smoothing

# Blink-rate health band (healthy focused reading = 10-25 per minute)
BLINK_RATE_LOW = 10.0
BLINK_RATE_HIGH = 25.0
BLINK_RATE_STARE_MIN = 5.0         # below = staring/hyperfocus

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "reports"
LOG_DIR.mkdir(exist_ok=True)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def draw_text(frame, text, position, color=(255, 255, 255), scale=0.6):
    cv2.putText(
        frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 2, cv2.LINE_AA
    )


def draw_focus_bar(frame, focus_pct, x, y, w, h):
    """Draw a horizontal focus meter."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    filled = int(w * (focus_pct / 100.0))
    # Color by level
    if focus_pct >= 70:
        color = (0, 200, 0)
    elif focus_pct >= 40:
        color = (0, 200, 255)
    else:
        color = (0, 0, 220)
    cv2.rectangle(frame, (x, y), (x + filled, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
    cv2.putText(
        frame, f"FOCUS {focus_pct:.0f}%",
        (x + 8, y + h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )


def get_gaze_direction(right_iris, left_iris):
    if right_iris is None or left_iris is None:
        return "UNKNOWN"
    # Iris positions are typically 2-tuples: (x, y) or arrays.
    # We take the x-component only.
    try:
        rx = right_iris[0] if hasattr(right_iris, "__len__") else right_iris
        lx = left_iris[0] if hasattr(left_iris, "__len__") else left_iris
    except Exception:
        return "UNKNOWN"
    average_iris = (float(rx) + float(lx)) / 2.0
    if average_iris < GAZE_LEFT_LIMIT:
        return "IMAGE LEFT"
    elif average_iris > GAZE_RIGHT_LIMIT:
        return "IMAGE RIGHT"
    return "CENTER"


def gaze_score(gaze_direction):
    """1.0 for center, lower as gaze deviates."""
    return {"CENTER": 1.0, "IMAGE LEFT": 0.15,
            "IMAGE RIGHT": 0.15, "UNKNOWN": 0.5}.get(gaze_direction, 0.5)


def head_score(head_direction):
    """1.0 forward, reduced for turns."""
    if "TURNED" in head_direction:
        return 0.15
    if "TILTED" in head_direction:
        return 0.60
    if head_direction == "UNKNOWN":
        return 0.5
    return 1.0


def eyes_open_score(eye_state, prolonged):
    """1.0 open, 0.0 closed; slightly reduced when prolonged closure."""
    if eye_state == "PROLONGED CLOSURE":
        return 0.0
    if eye_state == "CLOSED":
        return 0.2
    return 1.0


def blink_rate_score(rate_per_min):
    """
    Healthy focused reading: BLINK_RATE_LOW..BLINK_RATE_HIGH.
    Very low = staring/dissociated (not always bad, but a signal).
    Very high = fatigue / irritation.
    """
    if BLINK_RATE_LOW <= rate_per_min <= BLINK_RATE_HIGH:
        return 1.0
    if rate_per_min < BLINK_RATE_STARE_MIN:
        # Staring. Not zero — might be reading. Give partial credit.
        return 0.5
    if rate_per_min < BLINK_RATE_LOW:
        # Between stare-min and low: scale up
        t = (rate_per_min - BLINK_RATE_STARE_MIN) / (BLINK_RATE_LOW - BLINK_RATE_STARE_MIN)
        return 0.5 + 0.5 * t
    # Above high: fatigue rising
    excess = rate_per_min - BLINK_RATE_HIGH
    return max(0.3, 1.0 - excess / 60.0)


def classify_focus_state(focus_pct, gaze, head, eye_state, blink_rate):
    """Human-readable label based on combinations."""
    if eye_state == "PROLONGED CLOSURE":
        return "DROWSY / ZONED OUT"
    if focus_pct >= 75:
        return "FOCUSED"
    if blink_rate < BLINK_RATE_STARE_MIN and gaze == "CENTER" and head == "FORWARD":
        return "STARING / HYPER-FOCUSED"
    if gaze != "CENTER" or "TURNED" in head:
        return "DISTRACTED"
    if blink_rate > 30:
        return "FATIGUED"
    if 40 <= focus_pct < 75:
        return "DRIFTING"
    return "DISTRACTED"


# ==========================================
# MAIN
# ==========================================

def process_webcam():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Face model not found: {MODEL_PATH}")

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Check CAMERA_INDEX.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Webcam connected successfully!")
    print("Starting facial behaviour analysis...")
    print("Press Q to exit.")

    # ==========================================
    # SESSION STATE
    # ==========================================
    session_start = time.monotonic()

    blink_count = 0
    prolonged_closures = 0
    total_closure_time = 0.0
    total_frames = 0
    face_detected_frames = 0

    eye_closed = False
    closed_frame_count = 0
    closure_start = None
    prolonged_recorded = False

    smoothed_ear = None
    smoothed_focus = None
    last_focus_pct = 0.0

    previous_time = time.perf_counter()
    fps = 0.0
    last_timestamp_ms = -1

    # ==========================================
    # CALIBRATION
    # ==========================================
    calibration_ear_samples = []
    calibration_start = None
    calibrated = False
    EAR_THRESHOLD = 0.20  # will be replaced by calibration

    # ==========================================
    # CSV LOG
    # ==========================================
    session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = LOG_DIR / f"focus_log_{session_tag}.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp", "elapsed_s", "raw_ear", "smoothed_ear",
        "eye_state", "blink_count", "blink_rate",
        "gaze", "head", "yaw", "pitch", "roll",
        "focus_pct", "focus_state"
    ])

    # ==========================================
    # MEDIAPIPE
    # ==========================================
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read webcam frame.")
                    break

                total_frames += 1
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = time.monotonic_ns() // 1_000_000
                timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                current_time = time.monotonic()

                # ==== default state every frame (avoids stale vars) ====
                eye_state = "UNKNOWN"
                gaze_direction = "UNKNOWN"
                head_direction = "UNKNOWN"
                yaw = pitch = roll = 0.0
                raw_ear = 0.0
                current_focus_pct = last_focus_pct
                focus_state = "NO FACE"

                if result.face_landmarks:
                    face_detected_frames += 1
                    landmarks = result.face_landmarks[0]

                    # ---- bounding box ----
                    x_values = [p.x for p in landmarks]
                    y_values = [p.y for p in landmarks]
                    x1 = max(0, int(min(x_values) * width))
                    y1 = max(0, int(min(y_values) * height))
                    x2 = min(width - 1, int(max(x_values) * width))
                    y2 = min(height - 1, int(max(y_values) * height))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # ---- eyes ----
                    draw_eye(frame, landmarks, RIGHT_EYE, width, height)
                    draw_eye(frame, landmarks, LEFT_EYE, width, height)

                    # ---- EAR ----
                    right_ear = calculate_ear(landmarks, RIGHT_EYE, width, height)
                    left_ear = calculate_ear(landmarks, LEFT_EYE, width, height)
                    raw_ear = (right_ear + left_ear) / 2.0

                    if smoothed_ear is None:
                        smoothed_ear = raw_ear
                    else:
                        smoothed_ear = (
                            SMOOTH_ALPHA * raw_ear
                            + (1 - SMOOTH_ALPHA) * smoothed_ear
                        )

                    # ---- calibration phase ----
                    if not calibrated:
                        if calibration_start is None:
                            calibration_start = current_time
                            print("\n>>> CALIBRATION: hold eyes OPEN, look at camera <<<")

                        calibration_ear_samples.append(raw_ear)
                        elapsed_cal = current_time - calibration_start

                        if elapsed_cal >= CALIBRATION_SECONDS and len(calibration_ear_samples) > 5:
                            arr = np.array(calibration_ear_samples)
                            # Discard lowest 10% as possible blinks
                            arr = np.sort(arr)
                            trim = max(1, int(len(arr) * 0.10))
                            open_ear_est = float(np.median(arr[trim:]))
                            # Estimate closed as ~40% of open (empirical)
                            closed_ear_est = open_ear_est * 0.40
                            EAR_THRESHOLD = closed_ear_est + EAR_CLOSED_FRACTION * (open_ear_est - closed_ear_est)
                            calibrated = True
                            print(f">>> CALIBRATED: open~{open_ear_est:.3f}, "
                                  f"closed~{closed_ear_est:.3f}, "
                                  f"threshold={EAR_THRESHOLD:.3f} <<<\n")

                        # During calibration, treat eyes as open
                        eye_state = "CALIBRATING"
                    else:
                        # ---- BLINK DETECTION ON RAW EAR ----
                        if raw_ear < EAR_THRESHOLD:
                            closed_frame_count += 1
                            if not eye_closed:
                                # Wait for MIN_CLOSED_FRAMES before declaring closed
                                if closed_frame_count >= MIN_CLOSED_FRAMES:
                                    eye_closed = True
                                    closure_start = current_time
                                    prolonged_recorded = False
                            if eye_closed:
                                closure_duration = current_time - closure_start
                                eye_state = "CLOSED"
                                if closure_duration >= PROLONGED_CLOSURE_DURATION:
                                    eye_state = "PROLONGED CLOSURE"
                                    if not prolonged_recorded:
                                        prolonged_closures += 1
                                        prolonged_recorded = True
                            else:
                                eye_state = "OPEN"  # too few frames to count yet
                        else:
                            if eye_closed and closure_start is not None:
                                closure_duration = current_time - closure_start
                                if closure_duration >= MIN_BLINK_DURATION:
                                    total_closure_time += closure_duration
                                    if (closure_duration <= MAX_BLINK_DURATION
                                            and not prolonged_recorded):
                                        blink_count += 1
                                eye_closed = False
                                closure_start = None
                                prolonged_recorded = False
                            closed_frame_count = 0
                            eye_state = "OPEN"

                    # ---- closure time for display ----
                    current_closure = 0.0
                    if eye_closed and closure_start is not None:
                        current_closure = current_time - closure_start
                    live_closure_total = total_closure_time + current_closure

                    # ---- iris / gaze ----
                    right_iris = calculate_iris_position(
                        landmarks, RIGHT_IRIS, RIGHT_EYE, width, height
                    )
                    left_iris = calculate_iris_position(
                        landmarks, LEFT_IRIS, LEFT_EYE, width, height
                    )

                    for iris_indices in (RIGHT_IRIS, LEFT_IRIS):
                        iris_points = [
                            landmark_to_pixel(landmarks[i], width, height)
                            for i in iris_indices
                        ]
                        centre = (
                            int(np.mean([p[0] for p in iris_points])),
                            int(np.mean([p[1] for p in iris_points])),
                        )
                        cv2.circle(frame, centre, 5, (255, 0, 255), -1)

                    gaze_direction = get_gaze_direction(right_iris, left_iris)

                    # ---- head pose ----
                    head_pose = estimate_head_pose(landmarks, width, height)
                    if head_pose is not None:
                        pitch, yaw, roll = head_pose
                        head_direction = "FORWARD"
                        if yaw > HEAD_YAW_THRESHOLD:
                            head_direction = "TURNED RIGHT"
                        elif yaw < -HEAD_YAW_THRESHOLD:
                            head_direction = "TURNED LEFT"
                        if abs(pitch) > HEAD_PITCH_THRESHOLD:
                            head_direction += " / TILTED"
                    else:
                        pitch = yaw = roll = 0.0
                        head_direction = "UNKNOWN"

                    # ---- blink rate ----
                    elapsed_minutes = max((current_time - session_start) / 60.0, 1 / 60.0)
                    blink_rate = blink_count / elapsed_minutes

                    # ---- FOCUS SCORE ----
                    if calibrated:
                        s_gaze = gaze_score(gaze_direction)
                        s_head = head_score(head_direction)
                        s_eyes = eyes_open_score(eye_state, prolonged_recorded)
                        s_blink = blink_rate_score(blink_rate)

                        focus_raw = (
                            FOCUS_WEIGHT_GAZE * s_gaze
                            + FOCUS_WEIGHT_HEAD * s_head
                            + FOCUS_WEIGHT_EYES_OPEN * s_eyes
                            + FOCUS_WEIGHT_BLINK_RATE * s_blink
                        ) * 100.0

                        if smoothed_focus is None:
                            smoothed_focus = focus_raw
                        else:
                            smoothed_focus = (
                                FOCUS_SMOOTH_ALPHA * focus_raw
                                + (1 - FOCUS_SMOOTH_ALPHA) * smoothed_focus
                            )
                        current_focus_pct = smoothed_focus
                        last_focus_pct = current_focus_pct
                        focus_state = classify_focus_state(
                            current_focus_pct, gaze_direction,
                            head_direction, eye_state, blink_rate
                        )
                    else:
                        current_focus_pct = 0.0
                        focus_state = "CALIBRATING"

                    # ---- display ----
                    draw_text(frame, "LIVE FACIAL BEHAVIOUR ANALYSIS",
                              (20, 35), (0, 255, 0), 0.7)
                    draw_text(frame, f"EAR: {smoothed_ear:.3f}  (thr {EAR_THRESHOLD:.3f})",
                              (20, 75), (0, 255, 255))
                    state_color = (0, 0, 255) if eye_closed else (0, 255, 0)
                    draw_text(frame, f"Eye State: {eye_state}", (20, 110), state_color)
                    draw_text(frame, f"Blink Count: {blink_count}", (20, 150))
                    draw_text(frame, f"Blink Rate: {blink_rate:.1f}/min", (20, 185))
                    draw_text(frame, f"Closure: {current_closure:.2f}s", (20, 220))
                    draw_text(frame, f"Prolonged Closures: {prolonged_closures}",
                              (20, 255), (0, 165, 255))
                    draw_text(frame, f"Total Closure: {live_closure_total:.2f}s", (20, 290))
                    draw_text(frame, f"Gaze: {gaze_direction}", (20, 325), (255, 0, 255))
                    draw_text(frame, f"Head: {head_direction}", (20, 360), (255, 255, 0))
                    draw_text(frame, f"Yaw: {yaw:.1f} Pitch: {pitch:.1f}", (20, 395), (255, 255, 0))

                    # focus bar (bottom-left)
                    draw_focus_bar(frame, current_focus_pct, 20, height - 60, 380, 30)
                    draw_text(frame, f"State: {focus_state}",
                              (20, height - 80), (255, 255, 255), 0.75)

                    # ---- CSV row ----
                    csv_writer.writerow([
                        datetime.now().isoformat(timespec="seconds"),
                        f"{current_time - session_start:.3f}",
                        f"{raw_ear:.4f}",
                        f"{(smoothed_ear or 0):.4f}",
                        eye_state,
                        blink_count,
                        f"{blink_rate:.2f}",
                        gaze_direction,
                        head_direction,
                        f"{yaw:.2f}", f"{pitch:.2f}", f"{roll:.2f}",
                        f"{current_focus_pct:.1f}",
                        focus_state,
                    ])

                else:
                    draw_text(frame, "NO FACE DETECTED", (20, 40), (0, 0, 255), 0.7)
                    eye_closed = False
                    closed_frame_count = 0
                    closure_start = None
                    prolonged_recorded = False
                    smoothed_ear = None
                    focus_state = "NO FACE"
                    draw_text(frame, f"State: {focus_state}",
                              (20, height - 80), (200, 200, 200), 0.75)

                # ---- FPS ----
                frame_time = time.perf_counter()
                elapsed = frame_time - previous_time
                previous_time = frame_time
                if elapsed > 0:
                    current_fps = 1.0 / elapsed
                    fps = current_fps if fps == 0 else 0.9 * fps + 0.1 * current_fps
                draw_text(frame, f"FPS: {fps:.1f}",
                          (width - 160, 35), (0, 255, 255), 0.7)

                cv2.imshow("AI Behaviour Monitor - Facial Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        try:
            csv_file.close()
        except Exception:
            pass

        session_duration = time.monotonic() - session_start
        blink_rate = blink_count / max(session_duration / 60.0, 1 / 60.0)
        face_detection_rate = (face_detected_frames / max(total_frames, 1)) * 100

        print("\n" + "=" * 45)
        print("FACIAL BEHAVIOUR SESSION REPORT")
        print("=" * 45)
        print(f"Session duration: {session_duration:.2f} seconds")
        print(f"Total frames: {total_frames}")
        print(f"Face detection rate: {face_detection_rate:.1f}%")
        print(f"Calibrated EAR threshold: {EAR_THRESHOLD:.3f}")
        print(f"Total blinks: {blink_count}")
        print(f"Blink rate: {blink_rate:.2f} per minute")
        print(f"Prolonged closures: {prolonged_closures}")
        print(f"Total eye closure time: {total_closure_time:.2f} seconds")
        print(f"Final focus score: {last_focus_pct:.1f}%")
        print(f"Log saved: {csv_path}")
        print("=" * 45)
        print("Webcam released successfully.")


if __name__ == "__main__":
    process_webcam()