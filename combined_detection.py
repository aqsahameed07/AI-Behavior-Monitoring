
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "phone_wallet_mouse_v1"
    / "weights"
    / "best.pt"
)

HAND_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "hand_landmarker"
    / "hand_landmarker.task"
)

IMAGE_PATH = BASE_DIR / "sample_phone.jpg"

OUTPUT_PATH = BASE_DIR / "combined_detection_result.jpg"

CONFIDENCE = 0.25

# Only the phone class is used for hand proximity.
PHONE_CLASS = "phone"

# BGR colors
CLASS_COLORS = {
    "phone": (255, 0, 0),
    "wallet": (0, 255, 0),
    "mouse": (255, 255, 0)
}

HAND_COLOR = (0, 255, 0)


# ==========================================
# CHECK FILES
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}"
    )

if not HAND_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MediaPipe hand model not found: {HAND_MODEL_PATH}"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Input image not found: {IMAGE_PATH}"
    )


# ==========================================
# LOAD YOLO MODEL
# ==========================================

print("Loading trained YOLO26 model...")

yolo = YOLO(str(MODEL_PATH))

print("Model classes:", yolo.names)


# ==========================================
# LOAD MEDIAPIPE CONFIGURATION
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)


# ==========================================
# READ IMAGE
# ==========================================

original_image = cv2.imread(str(IMAGE_PATH))

if original_image is None:
    raise ValueError("Could not read input image.")

output_image = original_image.copy()

height, width = original_image.shape[:2]

diagonal = np.sqrt(width ** 2 + height ** 2)

proximity_threshold = max(40, diagonal * 0.05)


# ==========================================
# YOLO OBJECT DETECTION
# ==========================================

print("\nRunning YOLO detection...")

results = yolo.predict(
    source=original_image,
    conf=CONFIDENCE,
    verbose=False
)[0]

phone_boxes = []

detected_objects = []

for box in results.boxes:

    class_id = int(box.cls[0])

    confidence = float(box.conf[0])

    class_name = str(
        yolo.names[class_id]
    ).lower().strip()

    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0].tolist()
    )

    color = CLASS_COLORS.get(
        class_name,
        (255, 255, 255)
    )

    label = f"{class_name.upper()} {confidence:.2f}"

    # Draw object bounding box
    cv2.rectangle(
        output_image,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    cv2.putText(
        output_image,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    detected_objects.append(class_name)

    print(
        f"Detected: {class_name} | "
        f"Confidence: {confidence:.2%}"
    )

    # Store phone boxes only
    if class_name == PHONE_CLASS:

        phone_boxes.append({
            "box": (x1, y1, x2, y2),
            "confidence": confidence
        })


# ==========================================
# MEDIAPIPE HAND DETECTION
# ==========================================

print("\nRunning MediaPipe hand detection...")

rgb_image = cv2.cvtColor(
    original_image,
    cv2.COLOR_BGR2RGB
)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)

hand_points = []

with HandLandmarker.create_from_options(
    hand_options
) as landmarker:

    hand_result = landmarker.detect(mp_image)

    if hand_result.hand_landmarks:

        for hand_index, hand in enumerate(
            hand_result.hand_landmarks
        ):

            points = []

            for landmark in hand:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))

                points.append((x, y))

                cv2.circle(
                    output_image,
                    (x, y),
                    3,
                    HAND_COLOR,
                    -1
                )

            hand_points.append(points)


# ==========================================
# HAND-TO-PHONE PROXIMITY
# ==========================================

print("\n========== DETECTION RESULTS ==========")

print("Objects detected:", detected_objects)
print("Phone candidates:", len(phone_boxes))
print("Hands detected:", len(hand_points))

if not phone_boxes:
    print("\nNo phone detected.")

if not hand_points:
    print("\nNo hands detected.")

for hand_index, points in enumerate(hand_points):

    for phone_index, phone in enumerate(phone_boxes):

        x1, y1, x2, y2 = phone["box"]

        distances = []

        for x, y in points:

            dx = max(x1 - x, 0, x - x2)
            dy = max(y1 - y, 0, y - y2)

            distance = np.sqrt(
                dx ** 2 + dy ** 2
            )

            distances.append(distance)

        nearest_distance = min(distances)

        near_phone = (
            nearest_distance <= proximity_threshold
        )

        print(
            f"\nHand {hand_index + 1} -> "
            f"Phone {phone_index + 1}"
        )

        print(
            f"Nearest distance: "
            f"{nearest_distance:.1f}px"
        )

        if near_phone:

            status = "HAND NEAR PHONE"
            color = (0, 255, 255)

            print("Possible phone interaction detected!")

        else:

            status = "HAND AWAY FROM PHONE"
            color = (0, 0, 255)

            print("Hand is away from phone.")

        cv2.putText(
            output_image,
            status,
            (20, 35 + hand_index * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )


# ==========================================
# SUMMARY
# ==========================================

print("\n========== SUMMARY ==========")

for class_name in ["phone", "wallet", "mouse"]:

    count = detected_objects.count(class_name)

    print(f"{class_name.capitalize()}: {count}")

print("Hands:", len(hand_points))

if phone_boxes and hand_points:
    print("Hand-phone proximity analysis completed.")
else:
    print("Insufficient detections for proximity analysis.")


# ==========================================
# SAVE AND DISPLAY OUTPUT
# ==========================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

cv2.imwrite(
    str(OUTPUT_PATH),
    output_image
)

print("\n================================")
print("Combined detection completed!")
print(f"Saved: {OUTPUT_PATH}")
print("================================")

cv2.imshow(
    "YOLO26 + MediaPipe Combined Detection",
    output_image
)

cv2.waitKey(0)
cv2.destroyAllWindows()
