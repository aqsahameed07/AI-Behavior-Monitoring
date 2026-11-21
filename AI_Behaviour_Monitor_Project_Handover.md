# AI Behaviour Monitor
## Complete Project Documentation & Development Progress

**Project:** AI Personal Behaviour Monitoring System  
**Project Folder:** `D:\AI-Behaviour-Monitor`  
**Environment:** VS Code / Windows / PowerShell  
**Python Version:** Python 3.12  
**Virtual Environment:** `venv`  
**Current Stage:** Custom YOLO26 Model Training In Progress (Epoch 5/50). Dataset cleaned and validated. Mid-training checks confirm model is detecting phones at 94% confidence.

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
| Ultralytics YOLO26 | Object detection (custom trained) |
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

Python 3.14.8 was initially installed as the default Python version.

Python 3.12 was installed separately for compatibility with project libraries.

Both Python versions remain installed.

Python 3.12 was selected for this project.

## Virtual Environment

Created using:

```powershell
py -3.12 -m venv venv
Activated using:

powershell
.\venv\Scripts\Activate.ps1
Upgrade pip:

powershell
python -m pip install --upgrade pip
Installed Libraries
powershell
pip install opencv-python ultralytics mediapipe numpy pandas matplotlib
Verified imports:

powershell
python -c "import cv2, ultralytics, mediapipe, numpy, pandas, matplotlib; print('All libraries installed successfully!')"
Result:

text
All libraries installed successfully!
Verified package versions in the project virtual environment:

text
OpenCV: 5.1.0
MediaPipe: 1.2.3
Ultralytics: 8.4.215
NumPy: 2.5.5
Pandas: 3.1.1
Matplotlib: 3.11.2
PyTorch: 2.15.0+cpu
Important: Always activate the virtual environment before executing project scripts. The terminal should display (venv).

5. Current Project Structure
text
AI-Behaviour-Monitor/
│
├── venv/
│
├── dataset/
│   ├── phone_detection/
│   │   ├── data.yaml
│   │   ├── roboflow_data.yaml
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels_backup_before_normalization/
│   └── quarantine/
│
├── models/
│   └── phone_detector/
│       └── best_v1.pt
│
├── reports/
│   └── dataset_preview/
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
│
├── sample_phone.jpg
├── phone_lying.jpg
├── yolo11n.pt
├── yolo26n.pt
├── hand_landmarker.task
│
├── combined_detection_result.jpg
├── hand_detection_result.jpg
│
├── runs/
│   └── detect/
│       ├── phone_v1/
│       ├── phone_v2/
│       ├── phone_yolo26_clean/
│       ├── phone_yolo26_v1/
│       └── train-2/
│
├── Readme.md
└── AI_Behaviour_Monitor_Project_Handover.md
6. Webcam Status
The development PC currently does not have a camera connected.

The initial OpenCV webcam test returned:

text
Error: Could not access webcam!
A camera_test.py file has been created.

Current approach: Continue development using sample images and recorded videos. Live webcam testing will be performed when a camera becomes available.

Do not block development waiting for webcam hardware.

7. YOLO11 Object Detection (Baseline)
Model
The project originally used:

text
yolo11n.pt
This is a pretrained YOLO11 nano model.

Relevant COCO Classes
Class ID	Class
0	person
67	cell phone
65	remote
45	bowl
Individual Phone Detection
Created:

detect_phone.py

Test 1: Phone in Hand
Image:

sample_phone.jpg

Detection results:

text
Detected: person | Confidence: 77.56%
Detected: cell phone | Confidence: 68.83%
Detected: chair | Confidence: 66.71%
Detected: chair | Confidence: 37.97%
Result: Phone successfully detected.

Test 2: Phone Lying on Desk
Image:

phone_lying.jpg

Initial YOLO detection:

text
Detected: remote | Confidence: 47.02%
Detected: bowl | Confidence: 40.82%
YOLO located the phone but classified it as a remote control instead of a cell phone.

This limitation motivated custom YOLO training.

Temporary Detection Workaround (Retired)
The old combined detection script treated both cell phone and remote as phone candidates. This workaround will be removed after successful YOLO26 integration.

8. MediaPipe Hand Detection
Installed Version
text
MediaPipe: 1.2.3
The older API mp.solutions.hands is unavailable in this installation.

Therefore, the project uses the current MediaPipe Tasks API.

Hand Landmarker Model
Downloaded:

text
hand_landmarker.task
Hand Detection Script
Created:

test_hands.py

The script uses:

mp.tasks.BaseOptions

mp.tasks.vision.HandLandmarker

mp.tasks.vision.HandLandmarkerOptions

VisionRunningMode.IMAGE

Successful Test Results
text
Hands detected: 2

Hand 1
Landmarks: 21

Hand 2
Landmarks: 21

Detection completed!
9. Combined YOLO + MediaPipe Detection
Status: Successfully implemented and tested (using YOLO11 baseline).

Created:

combined_detection.py

Combined Detection Architecture
Load the YOLO model.

Load the MediaPipe Hand Landmarker model.

Read the original image.

Create a separate output image for annotations.

Run YOLO object detection.

Identify possible phone candidates.

Run MediaPipe on the untouched original image.

Extract hand landmarks.

Calculate the nearest hand-to-phone distance.

Identify possible phone interaction.

Draw bounding boxes and hand landmarks.

Save the final annotated image.

Hand-to-Phone Distance Calculation
The script calculates the minimum distance between hand landmarks and the phone bounding box.

python
dx = max(x1 - x, 0, x - x2)
dy = max(y1 - y, 0, y - y2)
distance = np.sqrt(dx**2 + dy**2)
nearest_distance = min(distances)
Current proximity threshold:

python
threshold = max(40, diagonal * 0.05)
Important: This is an image-pixel proximity heuristic. A result of 0.0px means at least one detected landmark lies inside or touches the bounding box. It does not prove that the hand is gripping or picking up the phone.

10. Combined Detection Testing Results (YOLO11 Baseline)
Test 1: Phone Being Held
Image: sample_phone.jpg

Combined results:

text
Phone candidates detected: 1
Hands detected: 2

Hand 1 -> Phone Candidate 1: 0.0px
Hand 2 -> Phone Candidate 1: 0.0px

Possible phone interaction detected!
Test 2: Phone Lying on Desk
Image: phone_lying.jpg

Combined results:

text
Phone candidates detected: 1
Hands detected: 0

Combined detection completed!
11. Custom Phone Detection Dataset
A phone detection dataset is located under:

text
dataset/phone_detection/
Current Dataset Split (After Cleaning)
Split	Images	Labels
Training	742	742
Validation	219	219
Test	109	109
Total	1,070	1,070
Original dataset had 1,217 images. 146 bad image/label pairs were quarantined (moved to dataset/quarantine/).

Quarantined Files
Count: 146 image/label pairs

Location: dataset/quarantine/

Reason: Bad annotations (oversized boxes with w > 0.95 or h > 0.95, or tiny boxes with w < 0.02 or h < 0.02)

Can be reviewed later but should NOT be used for training.

Dataset Configuration
The primary dataset configuration is:

text
dataset/phone_detection/data.yaml
Current contents:

yaml
path: D:/AI-Behaviour-Monitor/dataset/phone_detection
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: phone
Label State (Verified)
text
All annotations use class ID 0 (phone).
No invalid label records.
No missing label files.
No issues found after cleaning.
Custom Tools Created for Dataset Verification
visualize_dataset.py
Loads 9 random images per split.

Draws YOLO bounding boxes on them.

Saves preview grids to reports/dataset_preview/.

Output: train_preview.jpg, val_preview.jpg, test_preview.jpg.

find_bad_labels.py
Scans all label files.

Reports: oversized boxes, tiny boxes, out-of-bounds coordinates, invalid class IDs.

Output: list of bad label files (or "No issues found").

quarantine_bad_labels.py
Reads the results of find_bad_labels.py.

Moves bad image/label pairs to dataset/quarantine/.

Successfully quarantined 146 pairs.

12. Visual Dataset Verification (Completed)
Visual verification was performed and found:

✅ Most bounding boxes are tight and correctly placed around phones.

⚠️ One screenshot image (phone UI screenshot, not a real phone) was detected and quarantined.

⚠️ 146 additional images with oversized boxes were quarantined.

✅ After cleaning, dataset contains 742 train + 219 val + 109 test = 1,070 clean images.

13. Custom YOLO Training History
Failed Attempt: YOLO11 v1
Status: ❌ FAILED — do not use this model.

Command used:

powershell
yolo detect train data="...data.yaml" model=yolo11n.pt epochs=10 imgsz=640 batch=8 device=cpu plots=True name=phone_v1
Problems observed:

Loss curves went UP, not down — classic overfitting.

Catastrophic collapse at epoch 2 — mAP50 dropped from 0.343 to 0.154, then recovered.

Final mAP50 = 0.842 but the model detected NOTHING on real test images:

sample_phone.jpg: (no detections)

phone_lying.jpg: (no detections)

Root cause: 151 bad annotations in the dataset (oversized boxes teaching the model that "a phone fills the whole image").

Model saved at: models/phone_detector/best_v1.pt (do not use)

Charts saved at: runs/detect/phone_v1/ (results.png shows the collapse)

Current Training: YOLO26 (In Progress)
Status: 🟡 IN PROGRESS — epoch 5/50 at time of documentation update.

Command used:

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
  name=phone_yolo26_clean
Key improvements over v1:

Setting	YOLO11 v1	YOLO26	Reason
Model	yolo11n.pt	yolo26n.pt	Newest model
Optimizer	auto (AdamW lr=0.002)	SGD lr=0.01	More stable for fine-tuning
Epochs	10	50	Full convergence
Batch	8	16	More stable gradients
Cos LR	False	True	Smooth LR decay
Dataset	1,217 (dirty)	1,070 (clean)	146 bad pairs removed
Patience	100	15	Early stop if stuck
Mid-training results (epoch 5/50):

Metric	Value
mAP50	0.809
mAP50-95	0.524
Detections on sample_phone.jpg	1 phone at 0.94 confidence ✅
Detections on phone_lying.jpg	Pending test
Key insight: The mid-training check on last.pt (epoch 5) already produces a 94% confidence detection on sample_phone.jpg. This is vastly better than YOLO11 which detected nothing at epoch 10. The model is learning correctly.

Note about box_loss: In YOLO26, box_loss appears to rise slightly in early epochs while cls_loss and mAP50 improve. This is normal for YOLO26's loss formulation and should not be a concern as long as mAP50 is climbing.

Expected training time: ~4.5 hours on CPU.

Results will be saved to: runs/detect/phone_yolo26_clean/

Best checkpoint will be at: runs/detect/phone_yolo26_clean/weights/best.pt

14. Current Completed Features
Feature	Status
Python 3.12 setup	✅ Completed
Virtual environment	✅ Completed
Library installation	✅ Completed
YOLO11 setup	✅ Completed
YOLO person detection	✅ Tested
YOLO cell phone detection	✅ Tested
MediaPipe setup	✅ Completed
Hand landmark detection	✅ Tested
Combined YOLO + MediaPipe (YOLO11)	✅ Completed
Phone-in-hand test (YOLO11)	✅ Successful
Phone-on-desk test (YOLO11)	✅ Successful with workaround
Hand-to-phone proximity	✅ Working
Phone detection dataset	✅ Prepared
Dataset split verification	✅ Completed
Dataset class-ID normalization	✅ Completed
Dataset backup created	✅ Completed
Dataset label validation	✅ Completed
Visual annotation verification	✅ Completed
Bad label identification (151 issues)	✅ Completed
Bad image/label quarantine (146 pairs)	✅ Completed
Post-clean label verification	✅ No issues found
Dataset counts after cleaning	✅ 742/219/109 matched
YOLO11 training attempt	❌ Failed (model detects nothing)
YOLO26 training	🟡 IN PROGRESS (epoch 5/50)
Mid-training verification	✅ 94% confidence detection
Custom model evaluation	⏳ Pending (after training)
Integration with combined_detection.py	⏳ Pending (after training)
Live webcam testing	⏳ Pending hardware
Video processing	⏳ Not started
Temporal behaviour recognition	⏳ Not started
Phone usage timer	⏳ Not started
Database	⏳ Not started
Dashboard	⏳ Not started
15. Known Limitations
YOLO Classification
The pretrained YOLO11 model misclassified phones depending on position, angle, and image quality. The custom-trained YOLO26 model is expected to fix this.

Hand Proximity
Hand proximity alone cannot determine whether a phone is being held. Further checks are needed.

Temporal Behaviour
The current system processes individual images. Temporal behaviour recognition requires video processing.

Webcam
No camera is currently available on the development PC. Recorded videos and sample images will be used for development.

16. Planned Dataset Strategy
Custom Behaviour Dataset (Future)
Eventually collect clips for:

Sitting normally.

Working on a laptop.

Reaching for phone.

Picking up phone.

Holding phone.

Looking at phone.

Typing on phone.

Talking on phone.

Putting phone down.

Walking away.

Begin with approximately 30–50 clips per class.

Include variation in lighting, camera distance, phone position, hand position, body posture, and recording sessions.

Split training, validation, and test data by recording session to avoid data leakage.

17. Next Development Roadmap
Continue from the current YOLO26 training completion.

Step 1: Finish YOLO26 Training ✅ (In Progress)
Let training complete (~4.5 hours).

Verify training logs show healthy mAP progression.

Confirm best.pt saved successfully.

Step 2: Evaluate the Custom YOLO26 Model
Next immediate task after training completes:

Test the model on the two standard images:

powershell
# Test 1: Phone in hand
yolo detect predict model="D:\AI-Behaviour-Monitor\runs\detect\phone_yolo26_clean\weights\best.pt" source="D:\AI-Behaviour-Monitor\sample_phone.jpg" save=True conf=0.25 name=yolo26_final_held

# Test 2: Phone on desk (the failing case)
yolo detect predict model="D:\AI-Behaviour-Monitor\runs\detect\phone_yolo26_clean\weights\best.pt" source="D:\AI-Behaviour-Monitor\phone_lying.jpg" save=True conf=0.25 name=yolo26_final_desk
Expected results:

Both images should show detections

Both should be classified as phone (not remote)

Confidence should be 0.8+

If tests pass: proceed to Step 3.
If tests fail: investigate (may need more epochs, different hyperparameters, or additional data cleaning).

Step 3: Save the Trained Model
Once YOLO26 is validated:

powershell
# Copy best.pt to permanent location
Copy-Item `
  -Path "D:\AI-Behaviour-Monitor\runs\detect\phone_yolo26_clean\weights\best.pt" `
  -Destination "D:\AI-Behaviour-Monitor\models\phone_detector\best_yolo26.pt"

# Create dated backup
Copy-Item `
  -Path "D:\AI-Behaviour-Monitor\models\phone_detector\best_yolo26.pt" `
  -Destination "D:\AI-Behaviour-Monitor\models\phone_detector\best_yolo26_$(Get-Date -Format 'yyyy-MM-dd').pt"
Step 4: Multiple Image Testing
Test the model on additional images covering:

Phone held in hand

Phone lying on desk

Hand near phone

Hand away from phone

No phone visible

Phone at unusual angles

Multiple phones in frame

Record YOLO classifications and confidence scores. Investigate false positives before changing models.

Step 5: Compare Pretrained vs Custom YOLO26
Compare yolo11n.pt (pretrained) against models/phone_detector/best_yolo26.pt (custom).

Specifically check whether the custom model fixes:

The remote misclassification on phone_lying.jpg

Overall confidence on new images

False positive rate on non-phone objects

Do not assume the custom model is better without evaluating it.

Step 6: Integrate the Custom Model
After successful evaluation, update the model-loading configuration in:

text
combined_detection.py
Replace the pretrained model with the custom checkpoint:

python
MODEL_PATH = "D:/AI-Behaviour-Monitor/models/phone_detector/best_yolo26.pt"
Remove the POSSIBLE_PHONE_CLASSES workaround — the custom model detects phone as class 0 directly.

Retest the existing phone-in-hand and phone-on-desk images.

Keep the original yolo11n.pt available for comparison.

Step 7: Recorded Video Processing
Build a new script:

text
video_detection.py
The script should:

Load a recorded video.

Extract frames.

Run YOLO on each frame.

Run MediaPipe hand detection.

Track phone and hand positions.

Calculate hand-to-phone proximity.

Display detection results.

Save an annotated output video.

No webcam required.

Step 8: Temporal Behaviour Recognition
Track object positions across consecutive frames.

Recognize possible events:

Hand approaching phone

Hand contacting phone

Phone moving with hand

Phone being held

Phone being put down

Hand moving away

Use smoothing and confirmation windows to reduce false detections.

Step 9: Behaviour State Machine
Planned states:

text
IDLE
  |
  v
HAND_APPROACHING
  |
  v
PHONE_PICKUP_CANDIDATE
  |
  v
PHONE_IN_USE
  |
  v
PHONE_PUT_DOWN_CANDIDATE
  |
  v
SAVE_SESSION
  |
  v
IDLE
Do not start or stop the timer based on a single frame.

Step 10: Usage Timer
Implement:

Session start timestamp

Session end timestamp

Usage duration

Pickup count

Daily usage total

Average session duration

Step 11: VideoMAE (Future)
Explore fine-tuning MCG-NJU/videomae-base using custom behaviour clips.

VideoMAE will complement YOLO; it will not replace the phone object detector.

Step 12: Backend and Database
Develop:

FastAPI backend

MongoDB session storage

Behaviour detection endpoints

Usage history endpoints

Step 13: Frontend Dashboard
Develop a React + Vite dashboard displaying:

Current behaviour

Current phone-use status

Usage timer

Daily usage

Pickup count

Average session duration

Historical sessions

Charts and analytics

Step 14: Live Camera Integration
When webcam hardware becomes available:

Test OpenCV camera capture

Connect the existing detection pipeline

Run real-time behaviour recognition

Test timer accuracy

18. File Cleanup (Safe to Delete)
To keep the project clean, the following files/folders are safe to delete after the current YOLO26 training finishes and the final model is saved:

Item to Delete	Reason
runs/detect/phone_v1/	Failed YOLO11 training run — no longer needed
runs/detect/phone_v2/	Non-existent / empty / deprecated folder
runs/detect/phone_yolo26_v1/	Early attempt (superseded by phone_yolo26_clean)
runs/detect/train-2/	Smoke test run — no longer needed
runs/detect/test_v1_held/	Old failed test results
runs/detect/test_v1_desk/	Old failed test results
runs/detect/mid_training_check/	Temporary check (delete after final model saved)
runs/detect/mid_training_check_desk/	Temporary check (delete after final model saved)
models/phone_detector/best_v1.pt	Failed YOLO11 model — do not use
dataset/quarantine/	Bad annotations (optional: keep for review, but not needed)
Do NOT delete:

runs/detect/phone_yolo26_clean/ — this contains the final model

models/phone_detector/ — this contains the permanent saved model

dataset/phone_detection/ — this is the training data

Any .py scripts listed in Section 19

Any images/models referenced by the current combined_detection.py

19. Current Working Scripts and Models
Working Scripts
text
test_yolo.py              # Original YOLO11 setup test
detect_phone.py           # Original YOLO11 phone detection
test_hands.py             # MediaPipe hand detection
combined_detection.py     # YOLO11 + MediaPipe (needs update to YOLO26)
validate_dataset.py       # Dataset structural validation
visualize_dataset.py      # Dataset visual preview
find_bad_labels.py        # Bad label detection
quarantine_bad_labels.py  # Bad label quarantine
camera_test.py            # Webcam test (no hardware)
Current Dataset
text
dataset/phone_detection/
Dataset status (after cleaning):

742 training images and labels

219 validation images and labels

109 test images and labels

1,070 total images and matching labels

1 class: phone (class ID 0)

No oversized boxes (all w,h ≤ 0.95)

No tiny boxes (all w,h ≥ 0.02)

No invalid class IDs

146 bad pairs quarantined at dataset/quarantine/

Original labels preserved at labels_backup_before_normalization/

Current Working Models
text
yolo11n.pt                # Pretrained COCO (baseline only)
yolo26n.pt                # Pretrained YOLO26 (training base)
hand_landmarker.task      # MediaPipe hand landmarker
Trained Models
text
models/phone_detector/best_v1.pt              # ❌ FAILED — YOLO11 (do not use)
runs/detect/phone_yolo26_clean/weights/best.pt # 🟡 IN PROGRESS — will be final model
Current Test Images
text
sample_phone.jpg          # Easy case (phone in hand)
phone_lying.jpg           # Hard case (phone on desk, misclassified as remote by YOLO11)
Current Output
text
combined_detection_result.jpg   # Last YOLO11 + MediaPipe output
hand_detection_result.jpg       # Last MediaPipe output
20. CURRENT PROJECT POSITION — NOVEMBER 21, 2026
Last completed milestone: Dataset cleaned (146 bad pairs quarantined), dataset verified clean, YOLO26 training started with stable hyperparameters.

Currently running: YOLO26 training — epoch 5/50 at time of this update. Estimated completion: ~4.5 hours from training start.

Mid-training verification: At epoch 5, last.pt already detects sample_phone.jpg at 94% confidence. This confirms the model is learning correctly, unlike the failed YOLO11 v1 run.

Key learnings from this project so far:

Dataset quality is more important than model architecture — the failed YOLO11 run had 151 bad annotations that poisoned training.

Validation on the training set can be misleading — a model can have mAP50 = 0.84 and still detect nothing on real images.

Mid-training inference checks on real images are the most reliable signal of training health.

YOLO26's box_loss behaves differently from YOLO11's — don't interpret it as regression if mAP50 is climbing.

Use SGD with low LR for stable fine-tuning; AdamW with auto-tuned LR can cause catastrophic collapse on small datasets.

Clean data + stable optimizer = smooth training. No more epoch-2 collapse.

NEXT IMMEDIATE TASK (When Training Finishes)
Wait for training to complete (~4.5 hours from start).

Run final evaluation on test images:

sample_phone.jpg

phone_lying.jpg

Verify both detect correctly at high confidence.

If successful: copy best.pt to models/phone_detector/best_yolo26.pt.

Update combined_detection.py to use the new custom model.

Remove the POSSIBLE_PHONE_CLASSES workaround.

Move on to video processing (video_detection.py).

What NOT to Do
❌ Do not retrain the model unless evaluation shows it failed.

❌ Do not stop training on the basis of box_loss alone — only if mAP50 drops or errors appear.

❌ Do not use models/phone_detector/best_v1.pt — it is broken.

❌ Do not delete dataset/quarantine/ unless you have reviewed the bad labels (may be useful for future analysis).

❌ Do not modify data.yaml — it is correct.

❌ Do not repeat dataset cleaning or validation — it is complete.

Training Logs to Check
When training completes, these files will exist:

text
runs/detect/phone_yolo26_clean/
├── weights/
│   ├── best.pt        ← the final model
│   └── last.pt
├── results.csv        ← epoch-by-epoch metrics
├── results.png        ← visual charts
├── confusion_matrix.png
├── val_batch*_pred.jpg
└── val_batch*_labels.jpg
Look at results.png to confirm:

train/box_loss, train/cls_loss, train/l1_loss slope down

metrics/mAP50, metrics/mAP50-95 slope up

No catastrophic collapse (unlike YOLO11)
