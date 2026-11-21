"""
webcam_diag.py — Diagnose webcam frame delivery.
Saves 5 consecutive frames so we can see what's really coming through.
"""

import cv2
import time
import numpy as np

CAMERA_INDEX = 0
OUTPUT_PREFIX = "D:/AI-Behaviour-Monitor/diag_frame_"

print(f"Opening webcam {CAMERA_INDEX} with CAP_DSHOW...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ FAILED to open webcam")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Discarding 30 warm-up frames (this takes ~1 second)...")
for i in range(30):
    ret, _ = cap.read()
    if i % 10 == 0:
        print(f"  warmup {i}/30: ret={ret}")

print("\nSaving 5 consecutive frames...")
for i in range(5):
    ret, frame = cap.read()
    if not ret:
        print(f"  frame {i}: FAILED")
        continue

    mean_brightness = frame.mean()
    max_val = frame.max()
    is_black = mean_brightness < 5 and max_val < 20

    filename = f"{OUTPUT_PREFIX}{i}.jpg"
    cv2.imwrite(filename, frame)

    print(f"  frame {i}: shape={frame.shape}, "
          f"mean_brightness={mean_brightness:.1f}, "
          f"max={max_val}, "
          f"BLACK={is_black}, "
          f"saved={filename}")

    time.sleep(0.2)

cap.release()
print("\nDone. Open diag_frame_0.jpg through diag_frame_4.jpg and check if they show real content.")