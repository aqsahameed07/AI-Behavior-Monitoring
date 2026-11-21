"""
webcam_snapshot.py — Save one frame and test YOLO on it directly.
"""

import cv2
from ultralytics import YOLO

MODEL_PATH = "D:/AI-Behaviour-Monitor/models/phone_detector/best_yolo26.pt"
OUTPUT_IMAGE = "D:/AI-Behaviour-Monitor/webcam_snapshot.jpg"

print("Loading model...")
model = YOLO(MODEL_PATH)
print(f"Classes: {model.names}")

print("Opening webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit(1)

# Warm-up — grab 10 frames, discard them
for _ in range(10):
    cap.read()

print("Hold your phone up to the camera now... capturing in 2 seconds...")
import time
time.sleep(2)

ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Could not read frame.")
    exit(1)

cv2.imwrite(OUTPUT_IMAGE, frame)
print(f"Saved: {OUTPUT_IMAGE}")
print(f"Shape: {frame.shape}")

print("\nRunning YOLO at conf=0.05 (show ALL detections)...")
results = model(frame, conf=0.05, verbose=False)

count = 0
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_name = model.names[int(box.cls[0])]
        count += 1
        print(f"  {cls_name}  conf={conf:.3f}  box=({x1},{y1},{x2},{y2})")

print(f"\nTotal detections: {count}")
if count == 0:
    print("🚨 NO detections even at conf=0.05")