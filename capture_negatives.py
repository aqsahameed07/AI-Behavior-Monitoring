"""
capture_negatives.py — Capture negative examples (wallet, mouse, other items)
to teach YOLO that these are NOT phones.

Controls:
  SPACE = save current frame with current label
  W = switch to wallet mode
  M = switch to mouse mode
  Q = quit

Saves to:
  dataset/phone_detection/images/negatives/
  (label files will be created separately as empty .txt files)
"""

import cv2
import os
import time

OUTPUT_DIR = "D:/AI-Behaviour-Monitor/dataset/phone_detection/images/negatives"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CAMERA_INDEX = 0
TARGET_PER_CLASS = 30

modes = {"wallet": 0, "mouse": 0}
current_mode = "wallet"
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Warm up
for _ in range(20):
    cap.read()

print("=" * 60)
print("NEGATIVE CAPTURE TOOL")
print("=" * 60)
print("Modes:")
print("  Press W = capture a WALLET image")
print("  Press M = capture a MOUSE image")
print("  Press SPACE = save current frame with current mode")
print("  Press Q = quit")
print()
print(f"Targets: {TARGET_PER_CLASS} images per class")
print()
print("Show the object clearly in frame. Press SPACE when ready.")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    # Overlay
    display = frame.copy()
    status = f"Wallet: {modes['wallet']}/{TARGET_PER_CLASS}  |  Mouse: {modes['mouse']}/{TARGET_PER_CLASS}"
    cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)
    cv2.putText(display, f"Current mode: {current_mode.upper()}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, "W=wallet mode | M=mouse mode | SPACE=save | Q=quit",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Capture Negatives", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('w'):
        current_mode = "wallet"
        print(f"Switched to WALLET mode")
    elif key == ord('m'):
        current_mode = "mouse"
        print(f"Switched to MOUSE mode")
    elif key == 32:  # space
        if modes[current_mode] >= TARGET_PER_CLASS:
            print(f"Already have {TARGET_PER_CLASS} {current_mode} images. Switch mode or quit.")
            continue

        idx = modes[current_mode]
        filename = f"{current_mode}_{idx:03d}.jpg"
        path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(path, frame)
        modes[current_mode] += 1
        print(f"Saved: {path}  ({modes[current_mode]}/{TARGET_PER_CLASS})")

cap.release()
cv2.destroyAllWindows()
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Wallet images: {modes['wallet']}")
print(f"Mouse images:  {modes['mouse']}")
print(f"Total captured: {modes['wallet'] + modes['mouse']}")