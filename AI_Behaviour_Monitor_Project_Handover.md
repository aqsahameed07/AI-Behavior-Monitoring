# AI Behaviour Monitor
## Complete Project Documentation & Development Handover

**Project:** AI Personal Behaviour Monitoring System  
**Project Folder:** `D:\AI-Behaviour-Monitor\`  
**Environment:** VS Code / Windows / PowerShell  
**Python Version:** Python 3.12  
**Virtual Environment:** `venv`  
**Last Updated:** November 23, 2026  
**Current Stage:** YOLO26 three-class training completed. Real-world testing pending.

---

# 1. Project Overview

AI Behaviour Monitor is a computer vision-based personal behaviour monitoring system.

Its primary objective is to detect mobile phone interaction, recognize phone pickup and usage behaviour, calculate usage duration, and maintain a history of phone usage sessions.

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

**Important limitation:** A camera can estimate visible phone interaction but cannot reliably determine whether the phone screen is active or which application is being used.

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
| Ultralytics YOLO26 | Custom three-class object detection |
| MediaPipe Tasks | Hand landmark detection |
| NumPy | Mathematical calculations |
| Pandas | Data processing |
| Matplotlib | Data visualization |
| PyTorch | Deep learning |
| Hugging Face Transformers | VideoMAE (planned) |
| FastAPI | Backend API (planned) |
| MongoDB | Database (planned) |
| React + Vite | Frontend dashboard (planned) |

# 4. Development Environment

## Project Location

`D:\AI-Behaviour-Monitor\`

## Python Installation

Python 3.14.8 was initially installed as the default Python version.

Python 3.12 was installed separately for compatibility with project libraries. Both Python versions remain installed.

Python 3.12 is selected for this project.

## Virtual Environment

Created using:

```powershell
py -3.12 -m venv venv
```

Activated using:

```powershell
.\venv\Scripts\Activate.ps1
```

Installed libraries:

```powershell
pip install opencv-python ultralytics mediapipe numpy pandas matplotlib
```

Verified versions:

| Library | Version |
|---|---|
| OpenCV | 5.1.0 |
| MediaPipe | 1.2.3 |
| Ultralytics | 8.4.215 |
| NumPy | 2.5.5 |
| Pandas | 3.1.1 |
| Matplotlib | 3.11.2 |
| PyTorch | 2.15.0+cpu |

# 5. Current Project Structure

```text
AI-Behaviour-Monitor/
│
├── venv/
│
├── dataset/
│   ├── phone_detection/
│   │   ├── data.yaml
│   │   ├── images/
│   │   │   ├── train/ (802 images)
│   │   │   ├── val/   (234 images)
│   │   │   └── test/  (109 images)
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels_backup_before_normalization/
│   └── quarantine/
│
├── models/
│   └── phone_detector/
│       ├── best_v1.pt
│       ├── best_yolo26.pt
│       └── yolo26_last_pt.pt
│
├── runs/detect/
│   ├── phone_yolo26_final-2/
│   ├── phone_wallet_mouse_v1/
│   └── smoke_test_3class/
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
├── webcam_snapshot.py
├── webcam_diag.py
├── sample_phone.jpg
├── phone_lying.jpg
├── webcam_snapshot.jpg
├── diag_frame_0.jpg
├── diag_frame_1.jpg
├── diag_frame_2.jpg
├── diag_frame_3.jpg
├── diag_frame_4.jpg
│
├── yolo11n.pt
├── yolo26n.pt
├── hand_landmarker.task
│
├── Readme.md
└── AI_Behaviour_Monitor_Project_Handover.md
```

The final three-class model checkpoint currently exists at:

`runs/detect/phone_wallet_mouse_v1/weights/best.pt`

It still needs to be copied to:

`models/phone_detector/best_yolo26_3class.pt`

Do not assume that the final copy has already been created.

# 6. Webcam Status

Webcam is connected to the development PC.

Critical issue:

OpenCV's default MSMF backend returns black frames on this webcam.

DSHOW backend works but sometimes returns black frames during warmup.

Confirmed solution:

```python
cv2.VideoCapture(index, cv2.CAP_DSHOW)
```

Discard black frames until:

```python
frame.mean() > 15
```

In the main processing loop, skip frames where:

```python
frame.mean() < 10
```

`webcam_detection_v4.py` and `webcam_detection_v5.py` implement robust warmup.

Important findings:

- Webcam may return black frames for approximately 30–45 frames.
- DSHOW is preferred over MSMF.
- Resolution of 1280×720 is reliably supported.

# 7. Pretrained YOLO11 Baseline (Historical)

Model: `yolo11n.pt`

COCO pretrained model with 80 classes.

Relevant classes:

- Person (0)
- Cell phone (67)
- Remote (65)
- Bowl (45)

Results:

- `sample_phone.jpg`: cell phone detected at 68.83%.
- `phone_lying.jpg`: incorrectly classified as remote at 47.02%.

This motivated custom YOLO26 training.

# 8. MediaPipe Hand Detection (Working)

MediaPipe version: 1.2.3

Uses Tasks API.

Model:

`hand_landmarker.task`

Successfully tested:

- Two hands detected.
- 21 landmarks per hand.
- 42 total landmarks.

Used for hand-phone proximity filtering in `smart_detection.py`.

# 9. Combined YOLO + MediaPipe Detection

Current file:

`combined_detection.py`

Existing functionality:

1. Loads YOLO.
2. Loads MediaPipe.
3. Runs YOLO on the original image.
4. Runs MediaPipe on the original image, not an annotated image.
5. Computes minimum hand-to-phone-box distance.
6. Flags possible interaction if distance is below:

```python
max(40, diagonal * 0.05)
```

7. Saves annotated output.

**Current status:** Still uses the old YOLO11 baseline. It needs to be updated after the new three-class model passes real-world testing.

# 10. Custom Phone Dataset History

Original dataset:

- Train: 852 images.
- Validation: 243 images.
- Test: 122 images.
- Total: 1,217 images.

After quarantining problematic image/label pairs:

- Train: 742.
- Validation: 219.
- Test: 109.
- Total: 1,070.

Current expanded dataset:

- Train: 802 images.
- Validation: 234 images.
- Test: 109 images.
- Total: 1,145 images.

Class mapping:

| Class ID | Class | Training image distribution |
|---|---|---:|
| 0 | Phone | Approximately 742 |
| 1 | Wallet | 30 |
| 2 | Mouse | 30 |

Current annotation information:

- Phone: approximately 900 annotations across 742 files.
- Wallet: 35 annotations across 30 files.
- Mouse: 41 annotations across 30 files.

Wallet and mouse labels were remapped from Roboflow class 0 to classes 1 and 2.

Their filename prefixes are:

- `wallet_*`
- `mouse_*`

Important: Image counts and annotation counts are different. Some images contain multiple objects.

# 11. Dataset Cleaning Tools

## visualize_dataset.py

- Loads nine random images per split.
- Draws YOLO bounding boxes.
- Saves preview grids to `reports/dataset_preview/`.

## find_bad_labels.py

Scans label files for:

- Oversized boxes.
- Tiny boxes.
- Invalid class IDs.

Previously returned no issues after cleaning.

## quarantine_bad_labels.py

Moves bad image/label pairs to:

`dataset/quarantine/`

Successfully quarantined 146 pairs.

## capture_negatives.py

Captures webcam images for negative examples.

Controls:

- Spacebar: Save image.
- W/M: Switch modes.
- Q: Quit.

# 12. Training History

## Failed: YOLO11 v1

Dataset: Original 1,217 images.

Problem: Approximately 151 oversized bounding-box annotations.

Result:

- mAP50: 0.842.
- Failed real-world detection testing.

Cause: Incorrect annotations caused the model to learn unreliable bounding boxes.

Model:

`models/phone_detector/best_v1.pt`

Do not use.

## Successful: YOLO26 Phone-only

Training configuration:

| Setting | Value |
|---|---|
| Model | YOLO26n |
| Classes | 1 |
| Epochs | 50 |
| Image size | 640 |
| Batch | 16 |
| Device | CPU |
| Optimizer | SGD |

Results:

| Metric | Result |
|---|---:|
| mAP50 | 0.943 |
| mAP50-95 | 0.708 |
| Precision | 0.929 |
| Recall | 0.867 |
| Best epoch for mAP50-95 | 35 |
| Best mAP50 | 0.947 at epoch 36 |
| Training time | 3.98 hours |

Saved model:

`models/phone_detector/best_yolo26.pt`

This remains the phone-only fallback model.

## Completed: YOLO26 Three-Class Fine-Tuning

**STATUS: COMPLETED**

Objective:

Reduce false phone detections by explicitly training the model to distinguish phones, wallets and mice.

Class mapping:

- Class 0: Phone.
- Class 1: Wallet.
- Class 2: Mouse.

Training command:

```powershell
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
```

Training configuration:

| Setting | Value |
|---|---|
| Architecture | YOLO26n |
| Pretrained checkpoint | best_yolo26.pt |
| Classes | 3 |
| Training images | 802 |
| Validation images | 234 |
| Epochs | 20 |
| Image size | 640 |
| Batch | 16 |
| Device | CPU |
| Optimizer | SGD |
| Initial learning rate | 0.003 |
| Final LR factor | 0.01 |
| Cosine LR | Enabled |
| Patience | 8 |
| Training time | Approximately 1 hour 43 minutes |

Model architecture:

- YOLO26n.
- 2,504,970 parameters.
- Approximately 5.9 GFLOPs.
- Three detection classes.

Transferred pretrained weights:

708/708 items.

Training output:

`runs/detect/phone_wallet_mouse_v1/`

Best checkpoint:

`runs/detect/phone_wallet_mouse_v1/weights/best.pt`

### Overall validation results

| Metric | Result |
|---|---:|
| Precision | 84.6% |
| Recall | 85.8% |
| mAP50 | 85.7% |
| mAP50-95 | 73.1% |

### Class-wise results

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Phone | 89.3% | 90.8% | 94.1% | 69.6% |
| Wallet | 86.8% | 100% | 99.5% | 97.1% |
| Mouse | 77.7% | 66.7% | 63.6% | 52.6% |

### Performance assessment

**Phone**

Strong performance.

- mAP50: 94.1%.
- Recall: 90.8%.

This is encouraging for the primary project objective.

**Wallet**

Excellent validation scores.

However, there are only 10 validation instances. These results should not be treated as proof of generalization.

**Mouse**

Weakest class.

- mAP50: 63.6%.
- Recall: 66.7%.

Only six validation instances are available.

More diverse mouse images are needed.

### Training progression

| Epoch | mAP50 |
|---:|---:|
| 1 | 33.0% |
| 2 | 40.4% |
| 3 | 46.4% |
| 4 | 54.2% |
| 5 | 71.1% |
| 6 | 84.7% |
| 7 | 86.6% |
| 8 | 84.7% |
| 9 | 84.0% |
| 10 | 81.8% |
| 11 | 83.4% |
| 12 | 84.2% |
| 13 | 85.0% |
| 14 | 85.8% |
| 15 | 85.7% |
| 16 | 85.8% |
| 17 | 85.7% |
| 18 | 85.7% |
| 19 | 85.7% |
| 20 | 85.7% |

Assessment:

- Rapid improvement during the first seven epochs.
- Model stabilized after epoch 7.
- Best reported mAP50: 86.6%.
- Final reported mAP50: 85.7%.

No immediate retraining is necessary.

# 13. Webcam False Positive Problem and Solution

Original problem:

The phone-only YOLO26 model sometimes classified wallets and mice as phones.

Observed confidence:

Approximately 0.55–0.70.

Reason:

Phones, wallets and mice can share rectangular visual characteristics.

## Model-level solution

**COMPLETED**

Added wallet and mouse as separate classes.

The model can now learn the visual distinction instead of treating every rectangular object as a phone.

## Application-level solution

**PLANNED**

Implement additional filters in `smart_detection.py`.

### 1. Position filter

Consider object position in the camera frame.

Do not automatically reject every object in the bottom 25% of the frame because phones can legitimately be used on desks.

### 2. Persistence filter

Require detection across multiple consecutive frames.

Initial target:

Five consecutive frames.

### 3. Hand-proximity filter

Use MediaPipe hand landmarks.

Check distance between hands and phone bounding boxes.

Initial target:

150 pixels, subject to camera resolution and calibration.

### 4. Temporal interaction filter

Recognize:

- Hand approaching phone.
- Phone pickup.
- Phone holding.
- Phone put-down.

Important: Hand proximity alone cannot establish actual phone usage.

# 14. Webcam Diagnostic Files

| File | Purpose |
|---|---|
| webcam_detection_v4.py | Live detection with robust warmup |
| webcam_detection_v5.py | Live detection with aspect-ratio filter |
| smart_detection.py | Position, persistence and hand-proximity filters |
| webcam_snapshot.py | Single-frame diagnostic |
| webcam_diag.py | Five-frame webcam diagnostic |
| diag_frame_0.jpg to diag_frame_4.jpg | Diagnostic frames |

Key webcam requirements:

- Use `cv2.CAP_DSHOW`.
- Discard black frames during warmup.
- Wait for `frame.mean() > 15`.
- Skip frames with `frame.mean() < 10`.
- Use reliable 1280×720 resolution.

# 15. Known Limitations

## Model

- Phone-only model can confuse wallets and mice.
- Three-class model has completed training but still needs real-world testing.
- Mouse detection is the weakest class.
- Wallet and mouse validation samples are limited.
- Strong rectangular objects can still produce false detections.

## Hand proximity

- Hand proximity alone cannot establish phone usage.
- Must be combined with object detection and temporal filters.

## Webcam

- MSMF backend is unreliable on the development hardware.
- DSHOW requires warmup.
- 1280×720 is reliably supported.

## Behaviour recognition

Object detection cannot determine whether a person is actually using a phone.

The system needs temporal evidence and interaction analysis.

# 16. Next Development Roadmap

## Step 1: Complete three-class training

**COMPLETED**

YOLO26 three-class fine-tuning finished successfully.

## Step 2: Test the actual model on real images

**CURRENT NEXT TASK**

Do not modify existing scripts yet.

Run these commands separately in PowerShell.

### Test 1: Phone in hand

```powershell
yolo detect predict `
  model="D:\AI-Behaviour-Monitor\runs\detect\phone_wallet_mouse_v1\weights\best.pt" `
  source="D:\AI-Behaviour-Monitor\sample_phone.jpg" `
  save=True `
  conf=0.25 `
  name=TEST3_phone
```

### Test 2: Phone lying on desk

```powershell
yolo detect predict `
  model="D:\AI-Behaviour-Monitor\runs\detect\phone_wallet_mouse_v1\weights\best.pt" `
  source="D:\AI-Behaviour-Monitor\phone_lying.jpg" `
  save=True `
  conf=0.25 `
  name=TEST3_desk
```

### Test 3: Webcam snapshot

```powershell
yolo detect predict `
  model="D:\AI-Behaviour-Monitor\runs\detect\phone_wallet_mouse_v1\weights\best.pt" `
  source="D:\AI-Behaviour-Monitor\webcam_snapshot.jpg" `
  save=True `
  conf=0.25 `
  name=TEST3_webcam
```

Results will be saved under:

`runs/detect/`

Verify:

- Phone is classified as phone.
- Wallet is classified as wallet.
- Mouse is classified as mouse.
- Wallets and mice are not incorrectly classified as phones.
- Partially visible phones are still detected.

Share the resulting images for inspection.

## Step 3: Test the webcam

After image testing:

- Configure webcam detection to use the three-class model.
- Test phone detection.
- Test wallet detection.
- Test mouse detection.
- Check for false positives.
- Check performance in different positions and lighting conditions.

Expected classes:

| Object | Expected classification |
|---|---|
| Phone | Phone |
| Wallet | Wallet |
| Mouse | Mouse |

## Step 4: Save the final three-class model

After successful verification:

```powershell
Copy-Item `
  -Path "D:\AI-Behaviour-Monitor\runs\detect\phone_wallet_mouse_v1\weights\best.pt" `
  -Destination "D:\AI-Behaviour-Monitor\models\phone_detector\best_yolo26_3class.pt" `
  -Force
```

Create Desktop backup:

```powershell
Copy-Item `
  -Path "D:\AI-Behaviour-Monitor\models\phone_detector\best_yolo26_3class.pt" `
  -Destination "$env:USERPROFILE\Desktop\best_yolo26_3class_BACKUP.pt" `
  -Force
```

Keep `best_yolo26.pt` as the phone-only fallback.

## Step 5: Update combined_detection.py

After testing, change the model path:

```python
MODEL_PATH = (
    "D:/AI-Behaviour-Monitor/models/phone_detector/"
    "best_yolo26_3class.pt"
)

CLASSES_OF_INTEREST = ["phone"]
```

The application should:

- Load the three-class YOLO26 model.
- Detect phone, wallet and mouse.
- Use phone detections for interaction analysis.
- Continue using MediaPipe for hand landmarks.
- Calculate hand-to-phone-box distance.
- Display possible interaction.

Do not remove wallet and mouse from the trained model.

## Step 6: Improve smart_detection.py

Integrate:

- YOLO26 three-class detection.
- MediaPipe hand landmarks.
- Hand proximity.
- Detection persistence.
- Object tracking.
- Temporal interaction logic.

## Step 7: Build video_detection.py

Requirements:

- Load recorded videos.
- Process frames sequentially.
- Run YOLO26 on each frame.
- Run MediaPipe hand detection.
- Track phone and hand positions.
- Identify pickup and put-down events.
- Save annotated output video.
- Record interaction timestamps.

## Step 8: Behaviour State Machine

Proposed states:

```text
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
```

Do not start or stop the timer based on a single frame.

## Step 9: Usage Timer and Session Management

Record:

- Session start time.
- Session end time.
- Total duration.
- Number of pickups.
- Daily usage.
- Average session duration.
- Number of interactions.

## Step 10: Backend

Develop FastAPI + MongoDB backend.

Planned functionality:

- Session creation.
- Session updates.
- Usage history.
- Daily statistics.
- Analytics.

## Step 11: Frontend Dashboard

Develop React + Vite dashboard.

Display:

- Current monitoring status.
- Phone usage timer.
- Daily usage.
- Session history.
- Usage charts.
- Interaction statistics.

# 17. Safe-to-Delete Files

Potentially removable after confirming they are no longer needed:

| Item | Reason |
|---|---|
| runs/detect/phone_v1/ | Failed YOLO11 run |
| runs/detect/phone_yolo26_v1/ | Superseded run |
| runs/detect/smoke_test_3class/ | Smoke test |
| runs/detect/smoke_test_negatives/ | Smoke test |
| models/phone_detector/best_v1.pt | Failed YOLO11 model |
| dataset/quarantine/ | Quarantined bad labels, optional to retain |
| webcam_snapshot.jpg | Diagnostic output |
| diag_frame_*.jpg | Diagnostic output |

Never delete:

- `runs/detect/phone_yolo26_final-2/`
- `runs/detect/phone_wallet_mouse_v1/`
- `models/phone_detector/best_yolo26.pt`
- `models/phone_detector/best_yolo26_3class.pt` once created.
- `dataset/phone_detection/`
- All required Python scripts.
- `hand_landmarker.task`

Do not delete anything until backups are confirmed.

# 18. Working Scripts and Models

## Scripts

| Script | Purpose |
|---|---|
| test_yolo.py | YOLO setup test |
| detect_phone.py | Phone detection |
| test_hands.py | MediaPipe hand detection |
| combined_detection.py | YOLO + MediaPipe, requires YOLO26 update |
| validate_dataset.py | Dataset validation |
| visualize_dataset.py | Dataset visualization |
| find_bad_labels.py | Bad label detection |
| quarantine_bad_labels.py | Bad label quarantine |
| capture_negatives.py | Negative image capture |
| webcam_detection_v4.py | Webcam detection and warmup |
| webcam_detection_v5.py | Webcam detection with aspect filtering |
| smart_detection.py | Filtered interaction detection |
| webcam_snapshot.py | Webcam snapshot |
| webcam_diag.py | Webcam diagnostic frames |

## Models

| Model | Purpose / Status |
|---|---|
| yolo11n.pt | Pretrained YOLO11 |
| yolo26n.pt | Pretrained YOLO26 |
| best_v1.pt | Failed YOLO11 |
| best_yolo26.pt | Successful phone-only model |
| yolo26_last_pt.pt | Phone-only checkpoint |
| phone_wallet_mouse_v1/weights/best.pt | Completed three-class model |
| best_yolo26_3class.pt | Pending final copy |
| hand_landmarker.task | MediaPipe hand model |

## Test Images

- `sample_phone.jpg`: Phone in hand.
- `phone_lying.jpg`: Phone lying on desk.
- `webcam_snapshot.jpg`: Webcam snapshot.

# 19. Current Project Position — November 23, 2026

## Last completed milestone

YOLO26 three-class fine-tuning.

The original phone-only YOLO26 model achieved 94.3% mAP50 but sometimes confused phones with other rectangular objects.

To address this, the dataset was expanded with wallet and mouse classes.

The three-class model was fine-tuned using the successful phone-only checkpoint.

### Current model results

| Metric | Result |
|---|---:|
| Overall Precision | 84.6% |
| Overall Recall | 85.8% |
| Overall mAP50 | 85.7% |
| Overall mAP50-95 | 73.1% |

Class-wise mAP50:

- Phone: 94.1%.
- Wallet: 99.5%.
- Mouse: 63.6%.

## Current progress checklist

- [x] Clean original dataset.
- [x] Train YOLO26 phone-only model.
- [x] Expand dataset with wallet and mouse classes.
- [x] Configure three-class data.yaml.
- [x] Complete three-class smoke test.
- [x] Complete 20-epoch fine-tuning.
- [x] Evaluate overall validation metrics.
- [x] Evaluate class-wise metrics.
- [ ] Test sample_phone.jpg.
- [ ] Test phone_lying.jpg.
- [ ] Test webcam_snapshot.jpg.
- [ ] Verify live webcam detection.
- [ ] Save final model as best_yolo26_3class.pt.
- [ ] Create model backup.
- [ ] Update combined_detection.py.
- [ ] Integrate YOLO26 into smart_detection.py.
- [ ] Develop video_detection.py.
- [ ] Implement behaviour state machine.
- [ ] Implement usage timer and session storage.
- [ ] Develop FastAPI + MongoDB backend.
- [ ] Develop React + Vite dashboard.

## Immediate next task

**Run the three YOLO prediction tests and inspect the resulting images.**

Do not retrain the model immediately.

Do not modify `combined_detection.py` yet.

Do not overwrite the phone-only model.

The next development milestone is real-world verification.

## Important Research Considerations

1. Dataset quality is more important than simply increasing model complexity.
2. High validation mAP50 does not guarantee real-world detection reliability.
3. Wallet and mouse validation samples are limited.
4. YOLO detects objects, not human intentions.
5. MediaPipe detects hand landmarks, not proof of phone usage.
6. Combining object detection, hand proximity and temporal behaviour is the central technical contribution.
7. A phone lying on a desk should not automatically be treated as active phone usage.
8. The final system should distinguish object presence from actual interaction.

## Critical Warnings

- Do not retrain the phone-only model unless testing reveals a genuine issue.
- Do not delete `phone_yolo26_final-2/`.
- Do not delete `phone_wallet_mouse_v1/`.
- Do not modify data.yaml without verifying class mappings.
- Do not add negative images to validation/test sets without appropriate annotations.
- Do not run Git operations that could interfere with project files or training.
- Keep backups of both trained models.
- Do not integrate the new model before real-world verification.

---

# 20. Instructions for the Next ChatGPT Conversation

Continue this project from Section 19.

**Current confirmed position:**

- YOLO26 phone-only model trained successfully.
- YOLO26 three-class model trained successfully.
- Three classes: phone, wallet and mouse.
- Three-class training completed for 20 epochs.
- Overall validation mAP50: 85.7%.
- Phone mAP50: 94.1%.
- Wallet mAP50: 99.5%.
- Mouse mAP50: 63.6%.
- Real-world prediction testing has not yet been completed.
- Final three-class model copy and backup are pending.
- Existing combined_detection.py still needs updating.

**Start with these three tasks:**

1. Guide me through running the three prediction commands.
2. Examine the resulting detection images and help evaluate false positives and missed detections.
3. After successful testing, proceed to webcam testing and model integration.

**Development instructions:**

- Work step by step.
- Explain technical concepts in beginner-friendly language.
- Give complete, paste-ready code when modifying Python files.
- Preserve existing working functionality.
- Do not invent project files or assume that a model has been copied.
- Ask me to share existing code before modifying it.
- Do not skip testing.
- Do not retrain unnecessarily.
- Do not jump ahead to the backend or dashboard.

**The immediate starting point is real-world testing of the completed YOLO26 three-class model.**