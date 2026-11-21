
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================

# IMAGE_PATH = "phone_lying.jpg"

IMAGE_PATH = "sample_phone.jpg"
PHONE_CONFIDENCE = 0.25

# Treat remote as a possible phone candidate
POSSIBLE_PHONE_CLASSES = ["cell phone", "remote"]

# ==========================================
# LOAD MODELS
# ==========================================

print("Loading YOLO...")
yolo = YOLO("yolo11n.pt")

print("Loading MediaPipe...")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

# ==========================================
# READ IMAGE
# ==========================================

original_image = cv2.imread(IMAGE_PATH)

if original_image is None:
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

output_image = original_image.copy()

height, width = original_image.shape[:2]
diagonal = np.sqrt(width**2 + height**2)

# ==========================================
# YOLO OBJECT DETECTION
# ==========================================

print("\nRunning YOLO detection...")

results = yolo.predict(
    original_image,
    conf=PHONE_CONFIDENCE,
    verbose=False
)[0]

phone_boxes = []

for box in results.boxes:

    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = yolo.names[class_id]

    print(
        f"Detected: {class_name} "
        f"| Confidence: {confidence:.2%}"
    )

    if class_name in POSSIBLE_PHONE_CLASSES:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        phone_boxes.append({
            "box": (x1, y1, x2, y2),
            "class": class_name,
            "confidence": confidence
        })

        # Differentiate actual phone class from possible phone
        if class_name == "cell phone":
            label = f"PHONE {confidence:.2f}"
            color = (255, 0, 0)
        else:
            label = f"POSSIBLE PHONE (REMOTE) {confidence:.2f}"
            color = (0, 165, 255)

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
            0.55,
            color,
            2
        )

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

with HandLandmarker.create_from_options(options) as landmarker:

    hand_result = landmarker.detect(mp_image)

    if hand_result.hand_landmarks:

        for hand_index, hand in enumerate(
            hand_result.hand_landmarks
        ):

            points = []

            for landmark in hand:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

                cv2.circle(
                    output_image,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

            hand_points.append(points)

# ==========================================
# HAND-TO-PHONE PROXIMITY
# ==========================================

print("\n========== DETECTION RESULTS ==========")

print("Phone candidates detected:", len(phone_boxes))
print("Hands detected:", len(hand_points))

threshold = max(40, diagonal * 0.05)

for hand_index, points in enumerate(hand_points):

    for phone_index, phone in enumerate(phone_boxes):

        x1, y1, x2, y2 = phone["box"]

        distances = []

        for x, y in points:

            dx = max(x1 - x, 0, x - x2)
            dy = max(y1 - y, 0, y - y2)

            distance = np.sqrt(
                dx**2 + dy**2
            )

            distances.append(distance)

        nearest_distance = min(distances)

        near_phone = nearest_distance <= threshold

        print(
            f"Hand {hand_index + 1} -> "
            f"Phone Candidate {phone_index + 1}: "
            f"{nearest_distance:.1f}px"
        )

        if near_phone:

            print("Possible phone interaction detected!")

            status = "HAND NEAR PHONE"
            color = (0, 255, 255)

        else:

            print("Hand is away from phone.")

            status = "HAND AWAY"
            color = (0, 0, 255)

        cv2.putText(
            output_image,
            status,
            (20, 40 + hand_index * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

# ==========================================
# SAVE OUTPUT
# ==========================================

cv2.imwrite(
    "combined_detection_result.jpg",
    output_image
)

print("\n================================")
print("Combined detection completed!")
print("Saved: combined_detection_result.jpg")
print("================================")
