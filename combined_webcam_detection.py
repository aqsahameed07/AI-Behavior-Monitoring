
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

# YOLO26 three-class model
YOLO_MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "phone_wallet_mouse_v1"
    / "weights"
    / "best.pt"
)

# MediaPipe hand model: support both locations
HAND_MODEL_CANDIDATES = [
    BASE_DIR / "hand_landmarker.task",
    BASE_DIR / "models" / "hand_landmarker" / "hand_landmarker.task",
]

HAND_MODEL_PATH = next(
    (path for path in HAND_MODEL_CANDIDATES if path.exists()),
    HAND_MODEL_CANDIDATES[0]
)

# Webcam
CAMERA_INDEX = 0
CAMERA_BACKEND = cv2.CAP_MSMF

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# YOLO detection
CONFIDENCE = 0.20
IMAGE_SIZE = 640

CLASS_CONFIDENCE = {
    "phone": 0.50,
    "wallet": 0.35,
    "mouse": 0.35,
}

# Hand-to-phone proximity
PROXIMITY_THRESHOLD = 150
PERSISTENCE_FRAMES = 5

WINDOW_NAME = "AI Behaviour Monitor - YOLO26"


# OpenCV colors use BGR
CLASS_COLORS = {
    "phone": (255, 0, 0),
    "wallet": (0, 255, 0),
    "mouse": (255, 255, 0),
}

HAND_COLOR = (0, 255, 0)
INTERACTION_COLOR = (0, 255, 255)


# ==========================================
# CHECK REQUIRED FILES
# ==========================================

def check_files():

    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found:\n{YOLO_MODEL_PATH}"
        )

    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe hand model not found.\n"
            f"Checked locations:\n"
            f"{HAND_MODEL_CANDIDATES[0]}\n"
            f"{HAND_MODEL_CANDIDATES[1]}"
        )


# ==========================================
# LOAD YOLO MODEL
# ==========================================

def load_yolo():

    print("\nLoading YOLO26 three-class model...")

    model = YOLO(str(YOLO_MODEL_PATH))

    names = model.names

    class_names = {
        str(name).lower().strip()
        for name in names.values()
    } if isinstance(names, dict) else {
        str(name).lower().strip()
        for name in names
    }

    required_classes = {"phone", "wallet", "mouse"}

    if not required_classes.issubset(class_names):
        raise ValueError(
            f"Unexpected model classes: {names}\n"
            f"Expected: {required_classes}"
        )

    print("Model loaded successfully.")
    print("Model classes:", names)

    return model


# ==========================================
# MEDIAPIPE CONFIGURATION
# ==========================================

def create_hand_options():

    return mp.tasks.vision.HandLandmarkerOptions(

        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(HAND_MODEL_PATH)
        ),

        running_mode=mp.tasks.vision.RunningMode.VIDEO,

        num_hands=2,

        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )


# ==========================================
# WEBCAM INITIALIZATION
# ==========================================

def open_webcam():

    print(
        f"\nOpening webcam {CAMERA_INDEX} "
        f"using Media Foundation..."
    )

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        CAMERA_BACKEND
    )

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")
        cap.release()

        return None

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Webcam opened. Waiting for video frames...")

    # Wait for the camera to provide a valid frame.
    for attempt in range(50):

        ret, frame = cap.read()

        if ret and frame is not None and frame.size > 0:

            print(
                f"Webcam frame received successfully "
                f"(attempt {attempt + 1})."
            )

            print(
                "Resolution:",
                frame.shape[1],
                "x",
                frame.shape[0]
            )

            return cap

        time.sleep(0.1)

    print("ERROR: Webcam opened but no valid frames received.")

    cap.release()

    return None


# ==========================================
# YOLO OBJECT DETECTION
# ==========================================

def detect_objects(model, original_frame, display_frame):

    results = model.predict(
        source=original_frame,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device="cpu",
        verbose=False
    )[0]

    phone_boxes = []

    object_counts = {
        "phone": 0,
        "wallet": 0,
        "mouse": 0
    }

    for box in results.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = str(
            model.names[class_id]
        ).lower().strip()

        # Ignore unexpected classes
        if class_name not in CLASS_CONFIDENCE:
            continue

        # Apply individual class confidence threshold
        if confidence < CLASS_CONFIDENCE[class_name]:
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        color = CLASS_COLORS[class_name]

        object_counts[class_name] += 1

        label = (
            f"{class_name.upper()} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            display_frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        if class_name == "phone":
            phone_boxes.append(
                (x1, y1, x2, y2)
            )

    return phone_boxes, object_counts


# ==========================================
# MEDIAPIPE HAND DETECTION
# ==========================================

def detect_hands(
    landmarker,
    original_frame,
    display_frame,
    timestamp_ms
):

    height, width = original_frame.shape[:2]

    # MediaPipe receives the original frame,
    # not the annotated display frame.

    rgb_frame = cv2.cvtColor(
        original_frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    all_hand_points = []

    if result.hand_landmarks:

        for hand_index, hand in enumerate(
            result.hand_landmarks
        ):

            points = []

            for landmark in hand:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))

                points.append((x, y))

                cv2.circle(
                    display_frame,
                    (x, y),
                    3,
                    HAND_COLOR,
                    -1
                )

            all_hand_points.append(points)

            wrist_x, wrist_y = points[0]

            cv2.putText(
                display_frame,
                f"HAND {hand_index + 1}",
                (wrist_x, max(20, wrist_y - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                HAND_COLOR,
                2
            )

    return all_hand_points


# ==========================================
# HAND-TO-PHONE DISTANCE
# ==========================================

def calculate_hand_phone_distance(
    hand_points,
    phone_box
):

    x1, y1, x2, y2 = phone_box

    distances = []

    for x, y in hand_points:

        dx = max(x1 - x, 0, x - x2)
        dy = max(y1 - y, 0, y - y2)

        distance = np.sqrt(
            dx ** 2 + dy ** 2
        )

        distances.append(distance)

    if not distances:
        return float("inf")

    return min(distances)


# ==========================================
# PROXIMITY ANALYSIS
# ==========================================

def analyze_proximity(
    display_frame,
    hand_points,
    phone_boxes
):

    statuses = []

    nearest_distance = float("inf")

    hand_near_phone = False

    for hand_index, points in enumerate(hand_points):

        if not phone_boxes:

            status = (
                f"HAND {hand_index + 1}: "
                "NO PHONE DETECTED"
            )

            statuses.append(status)

            cv2.putText(
                display_frame,
                status,
                (10, 250 + hand_index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            continue

        hand_distance = float("inf")

        for phone_box in phone_boxes:

            distance = calculate_hand_phone_distance(
                points,
                phone_box
            )

            hand_distance = min(
                hand_distance,
                distance
            )

        nearest_distance = min(
            nearest_distance,
            hand_distance
        )

        near_phone = (
            hand_distance <= PROXIMITY_THRESHOLD
        )

        if near_phone:

            hand_near_phone = True

            status = (
                f"HAND {hand_index + 1}: "
                f"NEAR PHONE ({hand_distance:.0f}px)"
            )

            color = INTERACTION_COLOR

        else:

            status = (
                f"HAND {hand_index + 1}: "
                f"AWAY FROM PHONE ({hand_distance:.0f}px)"
            )

            color = (0, 0, 255)

        statuses.append(status)

        cv2.putText(
            display_frame,
            status,
            (10, 250 + hand_index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    return hand_near_phone, nearest_distance, statuses


# ==========================================
# LIVE DASHBOARD
# ==========================================

def draw_dashboard(
    frame,
    object_counts,
    hand_count,
    fps,
    interaction_streak,
    interaction_status
):

    cv2.putText(
        frame,
        "AI BEHAVIOUR MONITOR - YOLO26",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Phone: {object_counts['phone']}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        CLASS_COLORS["phone"],
        2
    )

    cv2.putText(
        frame,
        f"Wallet: {object_counts['wallet']}",
        (10, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        CLASS_COLORS["wallet"],
        2
    )

    cv2.putText(
        frame,
        f"Mouse: {object_counts['mouse']}",
        (10, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        CLASS_COLORS["mouse"],
        2
    )

    cv2.putText(
        frame,
        f"Hands: {hand_count}",
        (10, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        HAND_COLOR,
        2
    )

    cv2.putText(
        frame,
        f"Interaction frames: "
        f"{interaction_streak}/{PERSISTENCE_FRAMES}",
        (10, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    status_color = (
        INTERACTION_COLOR
        if interaction_streak >= PERSISTENCE_FRAMES
        else (255, 255, 255)
    )

    cv2.putText(
        frame,
        interaction_status,
        (10, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2
    )


# ==========================================
# MAIN
# ==========================================

def main():

    check_files()

    model = load_yolo()

    hand_options = create_hand_options()

    cap = open_webcam()

    if cap is None:
        return

    print("\n====================================")
    print("AI BEHAVIOUR MONITOR STARTED")
    print("YOLO26: PHONE / WALLET / MOUSE")
    print("MediaPipe: HAND LANDMARKS")
    print("Press Q to quit.")
    print("====================================\n")

    previous_time = time.perf_counter()

    fps_smooth = 0.0

    start_time = time.perf_counter()

    last_timestamp_ms = -1

    total_frames = 0

    interaction_streak = 0

    interaction_status = "NO INTERACTION"

    try:

        with mp.tasks.vision.HandLandmarker.create_from_options(
            hand_options
        ) as landmarker:

            while True:

                ret, original_frame = cap.read()

                if not ret or original_frame is None:
                    print("Failed to read webcam frame.")
                    break

                if original_frame.size == 0:
                    continue

                # Ignore completely dark frames
                if original_frame.mean() < 10:
                    interaction_streak = 0
                    interaction_status = "WAITING FOR CAMERA IMAGE"
                    continue

                # Preserve original camera frame
                display_frame = original_frame.copy()

                # ==================================
                # YOLO OBJECT DETECTION
                # ==================================

                phone_boxes, object_counts = detect_objects(
                    model,
                    original_frame,
                    display_frame
                )

                # ==================================
                # MEDIAPIPE HAND DETECTION
                # ==================================

                timestamp_ms = int(
                    (time.perf_counter() - start_time) * 1000
                )

                timestamp_ms = max(
                    timestamp_ms,
                    last_timestamp_ms + 1
                )

                last_timestamp_ms = timestamp_ms

                hand_points = detect_hands(
                    landmarker,
                    original_frame,
                    display_frame,
                    timestamp_ms
                )

                # ==================================
                # PROXIMITY ANALYSIS
                # ==================================

                hand_near_phone, nearest_distance, statuses = (
                    analyze_proximity(
                        display_frame,
                        hand_points,
                        phone_boxes
                    )
                )

                # ==================================
                # TEMPORAL PERSISTENCE
                # ==================================

                if hand_near_phone:
                    interaction_streak += 1
                else:
                    interaction_streak = 0

                if not phone_boxes:

                    interaction_status = "NO PHONE DETECTED"

                elif not hand_points:

                    interaction_status = (
                        "PHONE DETECTED - NO HAND"
                    )

                elif interaction_streak >= PERSISTENCE_FRAMES:

                    interaction_status = (
                        "POSSIBLE HAND-PHONE INTERACTION"
                    )

                elif hand_near_phone:

                    interaction_status = (
                        "POSSIBLE INTERACTION - CONFIRMING"
                    )

                else:

                    interaction_status = (
                        "PHONE DETECTED - NO NEAR HAND"
                    )

                # ==================================
                # FPS
                # ==================================

                current_time = time.perf_counter()

                elapsed = current_time - previous_time

                previous_time = current_time

                current_fps = 1.0 / max(elapsed, 1e-6)

                if fps_smooth == 0:
                    fps_smooth = current_fps
                else:
                    fps_smooth = (
                        0.9 * fps_smooth
                        + 0.1 * current_fps
                    )

                total_frames += 1

                # ==================================
                # DASHBOARD
                # ==================================

                draw_dashboard(
                    display_frame,
                    object_counts,
                    len(hand_points),
                    fps_smooth,
                    interaction_streak,
                    interaction_status
                )

                # ==================================
                # DISPLAY
                # ==================================

                cv2.imshow(
                    WINDOW_NAME,
                    display_frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

    except KeyboardInterrupt:

        print("\nDetection stopped by user.")

    finally:

        cap.release()

        cv2.destroyAllWindows()

        print("\n========== SESSION SUMMARY ==========")
        print(f"Frames processed: {total_frames}")
        print("YOLO model:", YOLO_MODEL_PATH.name)
        print("Webcam released.")
        print("Detection session completed.")


if __name__ == "__main__":
    main()
