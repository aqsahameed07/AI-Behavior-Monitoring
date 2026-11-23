# AI Behaviour Monitor
## Complete Project Documentation & Development Progress

**Project:** AI Personal Behaviour Monitoring System  
**Project Folder:** `D:\AI-Behaviour-Monitor`  
**Environment:** VS Code / Windows / PowerShell  
**Python Version:** Python 3.12  
**Virtual Environment:** `venv`  
**Current Stage:** Custom YOLO26 3-class model (phone + wallet + mouse) — fine-tuning IN PROGRESS.

---

# 1. Project Overview

The AI Behaviour Monitor is a computer vision-based personal behaviour monitoring system.

Its primary goal is to detect mobile phone interaction, recognize phone pickup and usage behaviour, calculate usage duration, and maintain a history of phone usage sessions.

The system will eventually monitor behaviours such as:

- Sitting normally.
- Working on a laptop.
- Reaching for a phone.
- Picking up a phone.
- Holding a phone.
- Looking at a phone.
- Typing on a phone.
- Talking on a phone.
- Putting a phone down.
- Walking away.

The initial focus is mobile phone usage detection.

**Important limitation:** A camera can estimate visible phone interaction, but cannot reliably determine whether the phone screen is active or which application is being used.

# 2. Main Project Objectives

1. Detect mobile phones using YOLO.
2. Detect hands and hand landmarks using MediaPipe.
3. Identify possible hand-to-phone interaction.
4. Track phone and hand movement across video frames.
5. Recognize phone pickup and put-down behaviour.
6. Start a timer when phone usage is confirmed.
7. Stop the timer when the phone is put down.
8. Save usage sessions.
9. Calculate daily phone usage statistics.
10. Display usage history and analytics through a dashboard.

# 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Main development language |
| OpenCV | Image and video processing |
| Ultralytics YOLO26 | Object detection (custom trained, 3 classes) |
| MediaPipe Tasks | Hand landmark detection |
| NumPy | Mathematical calculations |
| Pandas | Data processing |
| Matplotlib | Data visualization |
| PyTorch | Deep learning |
| Hugging Face Transformers | VideoMAE |
| FastAPI | Backend API |
| MongoDB | Database |
| React + Vite | Frontend dashboard |

# 4. Development Environment

## Project Location
`D:\AI-Behaviour-Monitor`

## Python Installation
Python 3.14.8 was initially installed as the default Python version. Python 3.12 was installed separately for compatibility with project libraries. Both Python versions remain installed. Python 3.12 was selected for this project.

## Virtual Environment
Created using:
```powershell
py -3.12 -m venv venv
Activated using:

powershell
.\venv\Scripts\Activate.ps1
Installed Libraries
powershell
pip install opencv-python ultralytics mediapipe numpy pandas matplotlib
Verified versions:

text
OpenCV: 5.1.0
MediaPipe: 1.2.3
Ultralytics: 8.4.215
NumPy: 2.5.5
Pandas: 3.1.1
Matplotlib: 3.11.2
PyTorch: 2.15.0+cpu
5. Current Project Structure
text
AI-Behaviour-Monitor/
│
├── venv/
│
├── dataset/
│   ├── phone_detection/
│   │   ├── data.yaml                       (nc: 3 — phone, wallet, mouse)
│   │   ├── images/
│   │   │   ├── train/                      (802 images: 742 phones + 30 wallets + 30 mice)
│   │   │   ├── val/                        (234 images: 219 phones + 10 wallets + 5 mice)
│   │   │   └── test/                       (109 images)
│   │   ├── labels/
│   │   │   ├── train/                      (802 labels)
│   │   │   ├── val/                        (234 labels)
│   │   │   └── test/                       (109 labels)
│   │   └── labels_backup_before_normalization/
│   └── quarantine/
│
├── models/
│   └── phone_detector/
│       ├── best_v1.pt                      (failed YOLO11 — do not use)
│       ├── best_yolo26.pt                  (phone-only model, mAP50 = 0.943)
│       └── yolo26_last_pt.pt               (epoch 50 phone-only)
│
├── runs/detect/
│   ├── phone_yolo26_final-2/               (completed phone-only 50-epoch run)
│   ├── phone_wallet_mouse_v1/              (CURRENT — 3-class fine-tuning)
│   └── smoke_test_3class/                  (smoke test)
│
├── reports/dataset_preview/
│
├── camera_test.py
├── test_yolo.py
├── detect_phone.py
├── test_hands.py
├── combined_detection.py
├── validate_dataset.py
├── visualize_dataset.py
├── find_bad_labels.py
├── quarantine_bad_labels.py
├── capture_negatives.py
├── webcam_detection_v4.py
├── webcam_detection_v5.py
├── smart_detection.py
│
├── sample_phone.jpg
├── phone_lying.jpg
├── webcam_snapshot.jpg
├── diag_frame_0.jpg ... diag_frame_4.jpg
├── yolo11n.pt
├── yolo26n.pt
├── hand_landmarker.task
│
├── Readme.md
└── AI_Behaviour_Monitor_Project_Handover.md
6. Webcam Status
Webcam IS connected on the development PC.

Critical issue: OpenCV's default MSMF backend returns black frames on this webcam. DSHOW backend works but sometimes returns black frames during warmup.

Confirmed fix: Use cv2.VideoCapture(index, cv2.CAP_DSHOW) and discard black frames until frame.mean() > 15.

webcam_detection_v4.py and webcam_detection_v5.py implement robust warmup that waits for real frames.

7. Pretrained YOLO11 Baseline (Historical)
Model: yolo11n.pt (COCO, 80 classes)

Relevant classes: person (0), cell phone (67), remote (65), bowl (45)

On sample_phone.jpg: cell phone 68.83%

On phone_lying.jpg: remote 47.02% (misclassification — motivated custom training)

8. MediaPipe Hand Detection (Working)
MediaPipe 1.2.3, uses Tasks API

Model: hand_landmarker.task

Tested: 2 hands, 21 landmarks each, 42 total

Used for hand-phone proximity filter in smart_detection.py

9. Combined YOLO + MediaPipe Detection
combined_detection.py:

Loads YOLO + MediaPipe

Runs YOLO on original image

Runs MediaPipe on original image (not annotated)

Computes minimum hand-to-phone-box distance

Flags "possible interaction" if distance < max(40, diagonal * 0.05)

Saves annotated output

Still using old YOLO11 baseline — needs update to 3-class model.

10. Custom Phone Dataset History
Original dataset: 852/243/122 (1,217 images)
After quarantine (146 bad pairs): 742/219/109 (1,070 images)
Current (with negatives added): 802/234/109

Class ID mapping in current dataset:

Class 0 = phone (~900 annotations across 742 files)

Class 1 = wallet (35 annotations across 30 files)

Class 2 = mouse (41 annotations across 30 files)

Important: Wallet and mouse labels were remapped from Roboflow's class 0 to classes 1 and 2 respectively. Their filename prefixes are wallet_* and mouse_*.

11. Dataset Cleaning Tools
visualize_dataset.py
Loads 9 random images per split

Draws YOLO bounding boxes

Saves preview grids to reports/dataset_preview/

find_bad_labels.py
Scans label files

Reports: oversized boxes, tiny boxes, invalid class IDs

Returns "No issues found" when clean

quarantine_bad_labels.py
Moves bad image/label pairs to dataset/quarantine/

Successfully quarantined 146 pairs

capture_negatives.py
Captures webcam images for negatives

Spacebar saves, W/M switch modes, Q quits

12. Training History
❌ Failed: YOLO11 v1
Dataset: original 1,217 images (dirty — 151 bad labels)

Result: mAP50 = 0.842 but detected NOTHING on real images

Cause: 151 oversized box annotations (box spanned entire image)

Model saved at: models/phone_detector/best_v1.pt (do not use)

✅ Success: YOLO26 phone-only
Command:

powershell
yolo detect train `
  data="D:\AI-Behaviour-Monitor\dataset\phone_detection\data.yaml" `
  model=yolo26n.pt `
  epochs=50 `
  imgsz=640 `
  batch=16 `
  device=cpu `
  optimizer=SGD `
  lr0=0.01 `
  lrf=0.01 `
  cos_lr=True `
  patience=15 `
  plots=True `
  name=phone_yolo26_final-2
Result:

Metric	Value
mAP50	0.943
mAP50-95	0.708
Precision	0.929
Recall	0.867
Best epoch (for mAP50-95)	35
Best mAP50	0.947 (epoch 36)
Training time	3.98 hours
Saved model: models/phone_detector/best_yolo26.pt (5.4 MB, epoch 35)

Backed up to:

$env:USERPROFILE\Desktop\best_yolo26_BACKUP.pt

D:\Phone_Dataset_Backup_YYYY-MM-DD\best_yolo26.pt

🟡 In Progress: YOLO26 3-class (phone + wallet + mouse)
Reason: Phone-only model has false positives on wallet and mouse (both are rectangular phone-shaped objects).

Smoke test passed: nc: 3 detected, 802 train images + 234 val images loaded, 1 epoch completed without errors.

Training command:

powershell
yolo detect train `
  data="D:\AI-Behaviour-Monitor\dataset\phone_detection\data.yaml" `
  model="D:\AI-Behaviour-Monitor\models\phone_detector\best_yolo26.pt" `
  epochs=20 `
  imgsz=640 `
  batch=16 `
  device=cpu `
  optimizer=SGD `
  lr0=0.003 `
  lrf=0.01 `
  cos_lr=True `
  patience=8 `
  plots=True `
  name=phone_wallet_mouse_v1
Expected output: runs/detect/phone_wallet_mouse_v1/weights/best.pt

Expected metrics after 20 epochs:

phone mAP50 ≥ 0.90

wallet mAP50 ≥ 0.75

mouse mAP50 ≥ 0.70

13. Webcam False Positive Problem (Diagnosed)
Observed false positives:

Wallet → detected as phone 0.55–0.70

Mouse → detected as phone 0.55–0.70

Cause: Both are rectangular, phone-shaped, and sometimes held near hands. The model learned "phone = rectangle near hand" and fires on them.

Two approaches to fix (both in progress):

Model-level (in progress): Add wallet and mouse as distinct classes (class 1 and 2). Model learns "wallet ≠ phone" explicitly.

Application-level (planned): After 3-class model is done, apply filters in smart_detection.py:

Position filter (reject bottom 25% of frame — desk objects)

Persistence filter (require 5+ consecutive frames)

Hand-proximity filter (MediaPipe hand within 150px of box)

14. Webcam Diagnostic Files
webcam_detection_v4.py — live detection with robust warmup + black frame handling

webcam_detection_v5.py — adds aspect-ratio filter (rejects non-phone-shaped boxes)

smart_detection.py — 3-layer filter (position, persistence, hand-proximity)

webcam_snapshot.py — saves one frame and runs YOLO with conf=0.05

webcam_diag.py — saves 5 consecutive frames with brightness checks

diag_frame_0.jpg .. diag_frame_4.jpg — saved diagnostic frames

Key finding: Webcam returns black frames until ~30-45 warmup frames. All detection scripts must:

Use cv2.CAP_DSHOW (not MSMF)

Discard frames until frame.mean() > 15

Skip any frame with frame.mean() < 10 in main loop

15. Known Limitations
Model Level
Phone-only model (0.943) fires on wallet/mouse — being fixed via 3-class training

Mouse/wallet detection depends on visual variety in training data

False positives possible on any strong rectangle

Hand Proximity
Hand proximity alone cannot determine whether phone is being held

Needs combined with model + temporal filters

Webcam
MSMF backend broken on this hardware

DSHOW works but needs 30+ frame warmup

Resolution limited to 1280×720 reliably

16. Next Development Roadmap
Step 1: Complete 3-class training ✅ (In Progress)
Command running now. Estimated 1.5 hours. Wait for 20 epochs completed.

Step 2: Test 3-class model
powershell
# Test on sample phone
yolo detect predict `
  model="D:\AI-Behaviour-Monitor\runs\detect\phone_wallet_mouse_v1\weights\best.pt" `
  source="D:\AI-Behaviour-Monitor\sample_phone.jpg" `
  save=True conf=0.25 name=TEST3_phone

# Test on desk phone
yolo detect predict `
  model="D:\AI-Behaviour-Monitor\runs\detect\phone_wallet_mouse_v1\weights\best.pt" `
  source="D:\AI-Behaviour-Monitor\phone_lying.jpg" `
  save=True conf=0.25 name=TEST3_desk
Expected: phone boxes at 0.85+ confidence.

Step 3: Webcam test with 3-class model
Update webcam_detection_v5.py MODEL_PATH to point to 3-class model. Run and verify:

Phone → phone box (green)

Wallet → wallet box (orange)

Mouse → mouse box (yellow)

Step 4: Save 3-class model
powershell
Copy-Item `
  -Path "D:\AI-Behaviour-Monitor\runs\detect\phone_wallet_mouse_v1\weights\best.pt" `
  -Destination "D:\AI-Behaviour-Monitor\models\phone_detector\best_yolo26_3class.pt" `
  -Force

Copy-Item `
  -Path "D:\AI-Behaviour-Monitor\models\phone_detector\best_yolo26_3class.pt" `
  -Destination "$env:USERPROFILE\Desktop\best_yolo26_3class_BACKUP.pt" `
  -Force
Keep best_yolo26.pt (phone-only) as fallback.

Step 5: Update combined_detection.py
Change to 3-class model:

python
MODEL_PATH = "D:/AI-Behaviour-Monitor/models/phone_detector/best_yolo26_3class.pt"
CLASSES_OF_INTEREST = ["phone"]  # ignore wallet/mouse
Test on sample_phone.jpg and phone_lying.jpg.

Step 6: Build smart_detection.py with app-layer filters
Already drafted. Filters:

Position filter — reject bottom 25% of frame (mouse on desk)

Persistence filter — require 5+ consecutive frames

Hand-proximity filter — require MediaPipe hand within 150px

Duration filter — reject sessions < 10 seconds (in behavior state machine)

Step 7: Build video_detection.py
Load recorded video

Run YOLO (3-class) + MediaPipe per frame

Apply filters

Save annotated output video

Step 8: Behavior State Machine
text
IDLE → HAND_APPROACHING → PHONE_PICKUP_CANDIDATE → PHONE_IN_USE
→ PHONE_PUT_DOWN_CANDIDATE → SAVE_SESSION → IDLE
Do not start/stop timer on single frame.

Step 9: Usage Timer + Database
Session start/end, duration, pickup count, daily total, average.

Step 10: Backend (FastAPI + MongoDB)
Step 11: Frontend (React + Vite dashboard)
17. Safe-to-Delete Files
Item	Reason
runs/detect/phone_v1/	Failed YOLO11
runs/detect/phone_yolo26_v1/	Superseded
runs/detect/smoke_test_3class/	Smoke test
runs/detect/smoke_test_negatives/	Smoke test
models/phone_detector/best_v1.pt	Failed YOLO11
dataset/quarantine/	Bad labels (optional to keep)
webcam_snapshot.jpg	Diagnostic output
diag_frame_*.jpg	Diagnostic output
Never delete:

runs/detect/phone_yolo26_final-2/ — phone-only model training

runs/detect/phone_wallet_mouse_v1/ — 3-class model training

models/phone_detector/best_yolo26.pt

models/phone_detector/best_yolo26_3class.pt (after created)

dataset/phone_detection/

All .py scripts

18. Working Scripts and Models
Scripts
text
test_yolo.py              # YOLO setup test
detect_phone.py           # Phone detection on image
test_hands.py             # MediaPipe hand detection
combined_detection.py     # YOLO11 + MediaPipe (needs update)
validate_dataset.py       # Dataset structural validation
visualize_dataset.py      # Dataset visual preview
find_bad_labels.py        # Bad label detection
quarantine_bad_labels.py  # Bad label quarantine
capture_negatives.py      # Webcam negative capture
webcam_detection_v4.py    # Live detection (robust warmup)
webcam_detection_v5.py    # Live detection + aspect filter
smart_detection.py        # 3-layer filtered detection
webcam_snapshot.py        # Single-frame diagnostic
webcam_diag.py            # Multi-frame diagnostic
Models
text
yolo11n.pt                            # Pretrained COCO
yolo26n.pt                            # Pretrained YOLO26
hand_landmarker.task                  # MediaPipe hand
models/phone_detector/best_v1.pt      # ❌ failed YOLO11
models/phone_detector/best_yolo26.pt  # ✅ phone-only (mAP50=0.943)
models/phone_detector/yolo26_last_pt.pt  # epoch 50
models/phone_detector/best_yolo26_3class.pt  # 🟡 pending 3-class training
Test Images
text
sample_phone.jpg    # Easy case (phone in hand)
phone_lying.jpg     # Hard case (desk phone — misclassified by YOLO11)
19. Current Project Position — November 23, 2026
Last completed milestone: Phone-only YOLO26 model trained (mAP50 = 0.943). Dataset expanded with 30 wallet + 30 mouse negatives (as separate classes). Smoke test passed. 3-class fine-tuning in progress.

Currently running: yolo detect train ... name=phone_wallet_mouse_v1 — 20 epochs, ~1.5 hours.

Mid-training status: Epoch ~5-8. Phone mAP50 should stay 0.90+, wallet/mouse starting to learn.

When training finishes:

Test on sample_phone.jpg and phone_lying.jpg — should show phone class at 0.85+

Webcam test — verify wallet shows as wallet (not phone)

Save to models/phone_detector/best_yolo26_3class.pt

Back up to Desktop + D:\Phone_Dataset_Backup_YYYY-MM-DD\

Update combined_detection.py to use 3-class model

Build smart_detection.py with 3-layer filters

Move to video_detection.py

Key learnings:

Dataset quality > model architecture

High val mAP50 ≠ works on real images

Mid-training inference checks are the real health signal

YOLO26 box_loss behaves differently — trust mAP50

SGD + low LR is stable for fine-tuning

Webcam requires DSHOW backend + warmup for real frames

YOLO will fire on any strong rectangle — app-layer filters needed

Adding negatives as separate classes works better than empty labels

Critical warnings:

❌ Do not retrain phone-only model unless it fails

❌ Do not delete phone_yolo26_final-2/ or phone_wallet_mouse_v1/

❌ Do not modify data.yaml without verifying nc: 3

❌ Do not add negatives to val/test splits (only train)

❌ Do not run git operations during training

❌ Do not close the training terminal

text

---

## How to Use This

**Save this entire content as `AI_Behaviour_Monitor_Project_Handover.md`** in `D:\AI-Behaviour-Monitor\`, replacing the old one.

When starting a new ChatGPT conversation, paste the first 100 lines and say: *"Continue the AI Behaviour Monitor project from Section 19. Here's the current state."*

The new assistant will know:
- Exactly what's in progress (3-class training)
- What to do next (test, save, integrate)
- What NOT to do (avoid data loss)
- Full context on all tools, models, and known issues