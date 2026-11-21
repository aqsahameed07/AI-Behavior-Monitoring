
from ultralytics import YOLO
import cv2
import os

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Image path
image_path = "sample_phone.jpg"

if not os.path.exists(image_path):
    print("Error: sample_phone.jpg not found!")
    exit()

print("Running phone detection...")

# Run detection
results = model.predict(
    source=image_path,
    conf=0.25,
    save=True
)

# Display detected objects
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        print(
            f"Detected: {class_name} | "
            f"Confidence: {confidence:.2%}"
        )

# Display detection result
print("\nDetection completed!")
print("Check the runs/detect folder for the output image.")
