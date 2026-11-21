
import cv2
import mediapipe as mp
import os

IMAGE_PATH = "sample_phone.jpg"
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Initialize MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

# Read image
image = cv2.imread(IMAGE_PATH)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)

print("Running hand detection...")

with HandLandmarker.create_from_options(options) as landmarker:

    result = landmarker.detect(mp_image)

    height, width, _ = image.shape

    if result.hand_landmarks:

        print(f"Hands detected: {len(result.hand_landmarks)}")

        for hand_index, hand in enumerate(result.hand_landmarks):

            print(f"\nHand {hand_index + 1}")
            print(f"Landmarks: {len(hand)}")

            for landmark in hand:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    image,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

    else:
        print("No hands detected in this image.")

# Save output
cv2.imwrite("hand_detection_result.jpg", image)

print("\nDetection completed!")
print("Output saved: hand_detection_result.jpg")
