# AI Behaviour Monitor

## Complete Project Documentation & Development Progress

**Project:** AI Personal Behaviour Monitoring System  

**Project Folder:** `D:\AI-Behaviour-Monitor`  

**Environment:** VS Code / Windows / PowerShell  

**Python Version:** Python 3.12  

**Virtual Environment:** `venv`  

**Current Stage:** Combined YOLO + MediaPipe Detection Tested; Custom Phone Dataset Normalized and Structurally Validated; Preparing for Visual Verification and Custom YOLO Training.

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

| Ultralytics YOLO11 | Object detection |

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

```

Activated using:

```powershell

.\venv\Scripts\Activate.ps1

```

Upgrade pip:

```powershell

python -m pip install --upgrade pip

```

## Installed Libraries

```powershell

pip install opencv-python ultralytics mediapipe numpy pandas matplotlib

```

Verified imports:

```powershell

python -c "import cv2, ultralytics, mediapipe, numpy, pandas, matplotlib; print('All libraries installed successfully!')"

```

Result:

```text

All libraries installed successfully!

```

Verified package versions in the project virtual environment:

```text

OpenCV: 5.1.0

MediaPipe: 1.2.3

Ultralytics: 8.4.215

NumPy: 2.5.5

Pandas: 3.1.1

Matplotlib: 3.11.2

PyTorch: 2.15.0+cpu

```

**Important:** Always activate the virtual environment before executing project scripts. The terminal should display `(venv)`.

# 5. Current Project Structure

```text

AI-Behaviour-Monitor/

│

├── venv/

│

├── dataset/

│   ├── raw/

│   ├── processed/

│   ├── phone_detection/

│   │   ├── data.yaml

│   │   ├── roboflow_data.yaml

│   │   ├── images/

│   │   │   ├── train/

│   │   │   ├── val/

│   │   │   └── test/

│   │   └── labels/

│   │       ├── train/

│   │       ├── val/

│   │       └── test/

│   │

│   └── action_recognition/

│

├── training/

│   ├── train_yolo.py

│   ├── train_videomae.py

│   ├── evaluate.py

│   └── preprocessing.py

│

├── models/

│   ├── phone_detector/

│   ├── behaviour_classifier/

│   └── pose_model/

│

├── backend/

│   ├── camera.py

│   ├── detection.py

│   ├── behaviour.py

│   ├── timer.py

│   ├── database.py

│   └── main.py

│

├── frontend/

│   └── React dashboard

│

├── reports/

│

├── camera_test.py

├── test_yolo.py

├── detect_phone.py

├── test_hands.py

├── combined_detection.py

├── validate_dataset.py

│

├── sample_phone.jpg

├── phone_lying.jpg

├── yolo11n.pt

├── hand_landmarker.task

│

├── combined_detection_result.jpg

├── hand_detection_result.jpg

│

├── README.md

└── requirements.txt

```

Note: Some directories and scripts are planned for future development and may not exist yet.

# 6. Webcam Status

The development PC currently does not have a camera connected.

The initial OpenCV webcam test returned:

```text

Error: Could not access webcam!

```

A `camera_test.py` file has been created.

**Current approach:** Continue development using sample images and recorded videos. Live webcam testing will be performed when a camera becomes available.

Do not block development waiting for webcam hardware.

# 7. YOLO11 Object Detection

## Model

The project uses:

```text

yolo11n.pt

```

This is a pretrained YOLO11 nano model.

The model was downloaded successfully through Ultralytics.

## Relevant COCO Classes

| Class ID | Class |

|---|---|

| 0 | person |

| 67 | cell phone |

| 65 | remote |

| 45 | bowl |

## YOLO Setup Test

Created:

`test_yolo.py`

Successfully tested model loading and class names.

Output:

```text

Loading YOLO model...

YOLO model loaded successfully!

Phone class: cell phone

Person class: person

YOLO setup completed!

```

## Individual Phone Detection

Created:

`detect_phone.py`

The script loads an image, performs YOLO detection, and saves an annotated output image.

Command:

```powershell

python detect_phone.py

```

### Test 1: Phone in Hand

Image:

`sample_phone.jpg`

Actual result:

```text

1 person

2 chairs

1 cell phone

```

Detection results:

```text

Detected: person | Confidence: 77.56%

Detected: cell phone | Confidence: 68.83%

Detected: chair | Confidence: 66.71%

Detected: chair | Confidence: 37.97%

```

**Result:** Phone successfully detected.

Output directory:

```text

runs/detect/predict/

```

### Test 2: Phone Lying on Desk

Image:

`phone_lying.jpg`

Initial YOLO detection:

```text

Detected: remote | Confidence: 47.02%

Detected: bowl | Confidence: 40.82%

```

YOLO located the phone but classified it as a remote control instead of a cell phone.

This demonstrates a limitation of the pretrained model with certain phone angles.

The issue is not a Python error.

## Temporary Detection Workaround

The combined detection script now treats both classes as possible phone candidates:

```python

POSSIBLE_PHONE_CLASSES = ["cell phone", "remote"]

```

Detection condition:

```python

if class_name in POSSIBLE_PHONE_CLASSES:

```

Labels:

- `PHONE` for YOLO's cell phone classification.

- `POSSIBLE PHONE (REMOTE)` for remote classification.

This workaround allows testing but can introduce false positives because actual remote controls may also be detected.

It is not a permanent phone classifier.

# 8. MediaPipe Hand Detection

## Installed Version

```text

MediaPipe: 1.2.3

```

The older API:

```python

mp.solutions.hands

```

is unavailable in this installation.

Therefore, the project uses the current MediaPipe Tasks API.

## Hand Landmarker Model

Downloaded:

```text

hand_landmarker.task

```

Download command:

```powershell

Invoke-WebRequest `

  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `

  -OutFile "hand_landmarker.task"

```

## Hand Detection Script

Created:

`test_hands.py`

The script uses:

- `mp.tasks.BaseOptions`

- `mp.tasks.vision.HandLandmarker`

- `mp.tasks.vision.HandLandmarkerOptions`

- `VisionRunningMode.IMAGE`

It detects hands, extracts landmarks, draws the landmarks, and saves:

```text

hand_detection_result.jpg

```

### Successful Test Results

```text

Hands detected: 2

Hand 1

Landmarks: 21

Hand 2

Landmarks: 21

Detection completed!

```

Total:

- 2 hands detected.

- 21 landmarks per hand.

- 42 landmarks in total.

**Result:** MediaPipe hand detection is working.

# 9. Combined YOLO + MediaPipe Detection

**Status: Successfully implemented and tested.**

Created:

`combined_detection.py`

The purpose of this script is to combine phone object detection and hand landmark detection on the same image.

## Combined Detection Architecture

1. Load the YOLO11 model.

2. Load the MediaPipe Hand Landmarker model.

3. Read the original image.

4. Create a separate output image for annotations.

5. Run YOLO object detection.

6. Identify possible phone candidates.

7. Run MediaPipe on the untouched original image.

8. Extract hand landmarks.

9. Calculate the nearest hand-to-phone distance.

10. Identify possible phone interaction.

11. Draw bounding boxes and hand landmarks.

12. Save the final annotated image.

## Important Image Processing Improvement

Initially, YOLO annotations were drawn before MediaPipe processed the image.

This was corrected.

Current approach:

```python

original_image = cv2.imread(IMAGE_PATH)

output_image = original_image.copy()

```

MediaPipe processes the original image:

```python

rgb_image = cv2.cvtColor(

    original_image,

    cv2.COLOR_BGR2RGB

)

```

Annotations are drawn on:

```python

output_image

```

This ensures that MediaPipe does not process the YOLO annotation overlays.

## Phone Detection Configuration

Current configuration:

```python

IMAGE_PATH = "sample_phone.jpg"

PHONE_CONFIDENCE = 0.25

POSSIBLE_PHONE_CLASSES = [

    "cell phone",

    "remote"

]

```

The image path can be changed to test different images.

## Hand-to-Phone Distance Calculation

The script calculates the minimum distance between hand landmarks and the phone bounding box.

```python

dx = max(x1 - x, 0, x - x2)

dy = max(y1 - y, 0, y - y2)

distance = np.sqrt(

    dx**2 + dy**2

)

```

The nearest distance is calculated using:

```python

nearest_distance = min(distances)

```

Current proximity threshold:

```python

threshold = max(40, diagonal * 0.05)

```

If the nearest distance is below the threshold:

```text

Possible phone interaction detected!

```

**Important:** This is an image-pixel proximity heuristic. A result of `0.0px` means at least one detected landmark lies inside or touches the bounding box. It does not prove that the hand is gripping or picking up the phone.

# 10. Combined Detection Testing Results

Two main test cases have been completed successfully.

## Test 1: Phone Being Held

Image:

`sample_phone.jpg`

YOLO output:

```text

Detected: person | Confidence: 77.56%

Detected: cell phone | Confidence: 68.83%

Detected: chair | Confidence: 66.71%

Detected: chair | Confidence: 37.97%

```

MediaPipe:

```text

Hands detected: 2

```

Combined results:

```text

========== DETECTION RESULTS ==========

Phone candidates detected: 1

Hands detected: 2

Hand 1 -> Phone Candidate 1: 0.0px

Possible phone interaction detected!

Hand 2 -> Phone Candidate 1: 0.0px

Possible phone interaction detected!

Combined detection completed!

Saved: combined_detection_result.jpg

```

Result:

- Phone detected.

- Both hands detected.

- Both hands are within the phone bounding box.

- Possible phone interaction detected.

- Annotated image saved successfully.

**Status: Successful test.**

## Test 2: Phone Lying on Desk

Image:

`phone_lying.jpg`

Initial result using only the `cell phone` class:

```text

Phones detected: 0

Hands detected: 0

```

YOLO classified the phone as a remote.

After implementing the temporary remote-class workaround:

```text

Detected: remote | Confidence: 47.02%

Detected: bowl | Confidence: 40.82%

```

Combined results:

```text

========== DETECTION RESULTS ==========

Phone candidates detected: 1

Hands detected: 0

Combined detection completed!

Saved: combined_detection_result.jpg

```

Result:

- Possible phone candidate detected.

- Remote classification successfully handled.

- No hands detected, as expected for this image.

- Annotated image saved successfully.

**Status: Successful workaround test.**

# 11. Custom Phone Detection Dataset

A phone detection dataset has been added under:

```text

dataset/phone_detection/

```

The dataset contains YOLO-format images and labels.

## Dataset Split

| Split | Images | Labels |

|---|---:|---:|

| Training | 852 | 852 |

| Validation | 243 | 243 |

| Test | 122 | 122 |

| **Total** | **1,217** | **1,217** |

Every image has a matching label file and no label file is empty, according to the completed structural validation.

## Dataset Normalization and Validation

Created:

`validate_dataset.py`

The script performs the following safety checks:

1. Confirms that all train, validation, and test label folders exist.

2. Confirms that label files contain exactly five fields.

3. Confirms that class IDs are valid.

4. Confirms that normalized coordinates are numeric and within the expected range.

5. Confirms that bounding-box width and height are greater than zero.

6. Creates a complete backup before changing any labels.

7. Restores the original labels automatically if a write error occurs.

The original labels are preserved under:

```text

dataset/phone_detection/labels_backup_before_normalization/

```

The backup contains the original split structure:

| Split | Original Label Files |

|---|---:|

| Training | 852 |

| Validation | 243 |

| Test | 122 |

All current phone annotations use class ID `0`.

Verified current annotation state:

```text

Class 0: 1,462 annotations

Class 1: 0 annotations

Unexpected class IDs: 0

Invalid label records: 0

Missing label files: 0

```

The normalization script preserves the original bounding-box coordinates and changes only the class ID.

## Dataset Configuration

The primary dataset configuration is:

```text

dataset/phone_detection/data.yaml

```

It declares:

```yaml

path: D:/AI-Behaviour-Monitor/dataset/phone_detection

train: images/train

val: images/val

test: images/test

nc: 1

names:

  0: phone

```

The actual file should be checked before training to ensure these settings are present.

## Label Verification

The initial label review found that the downloaded YOLO labels used class ID `1` for most phone annotations:

```text

Class 1: 1,461 annotations

Class 0: 1 annotation

```

This mismatch has now been corrected:

```text

All 1,462 annotations use class ID 0 (phone).

```

The original labels were backed up before normalization.

No label-format, bounding-box-range, or missing-label problems were found.

The dataset configuration now matches the normalized label files and is ready for the next custom YOLO validation/training step.

## Additional Dataset Metadata

`roboflow_data.yaml` contains exported Roboflow metadata and declares a two-class configuration:

```yaml

nc: 2

names: ['6', 'phone']

```

This file appears to describe the original export structure and should not be used for training until its class IDs are reconciled with `data.yaml` and the actual label files.

# 12. Latest PowerShell Verification — November 19, 2026

After completing dataset normalization and validation, an additional PowerShell command was executed to verify image and label counts.

The terminal was already inside the activated virtual environment:

```text

(venv) PS D:\AI-Behaviour-Monitor>

```

Command executed:

```powershell

$target = "D:\AI-Behaviour-Monitor\dataset\phone_detection"; Write-Host "TRAIN IMAGES:" (Get-ChildItem "$target\images\train" -File -ErrorAction SilentlyContinue).Count; Write-Host "TRAIN LABELS:" (Get-ChildItem "$target\labels\train" -File -ErrorAction SilentlyContinue).Count; Write-Host "VAL IMAGES:" (Get-ChildItem "$target\images\val" -File -ErrorAction SilentlyContinue).Count; Write-Host "VAL LABELS:" (Get-ChildItem "$target\labels\val" -File -ErrorAction SilentlyContinue).Count; Write-Host "TEST IMAGES:" (Get-ChildItem "$target\images\test" -File -ErrorAction SilentlyContinue).Count; Write-Host "TEST LABELS:" (Get-ChildItem "$target\labels\test" -File -ErrorAction SilentlyContinue).Count

```

Actual output:

```text

TRAIN IMAGES: 852

TRAIN LABELS: 852

VAL IMAGES: 243

VAL LABELS: 243

```

### Analysis of Latest Output

The latest PowerShell check confirms:

| Split | Images | Labels | Result |

|---|---:|---:|---|

| Training | 852 | 852 | Matched |

| Validation | 243 | 243 | Matched |

| Test | 122 | 122 | Previously verified; not printed in latest command |

Training and validation image/label counts match perfectly.

The test split was previously documented as 122 images and 122 labels through dataset validation, but the latest PowerShell output did not display those counts.

**Conclusion:** Training and validation folder counts are confirmed again. No additional normalization is required based on this output.

# 13. Dataset Visual Verification — Next Immediate Task

Structural validation confirms that labels have the correct format and class IDs. However, it does not prove that bounding boxes accurately surround the phones.

Before training, representative annotations should be inspected visually.

The next task is to create:

```text

visualize_dataset.py

```

Expected functionality:

1. Load random images from the training split.

2. Load their corresponding YOLO label files.

3. Convert normalized coordinates into image coordinates.

4. Draw bounding boxes around detected phones.

5. Repeat for validation and test images.

6. Save preview images in:

```text

reports/dataset_preview/

```

Expected outputs:

```text

train_preview.jpg

val_preview.jpg

test_preview.jpg

```

The previews should contain approximately nine randomly selected images from each split.

### What Must Be Checked

- Bounding boxes should surround the complete phone.

- Boxes should not cover an unnecessarily large portion of the image.

- Boxes should not be too small.

- The phone should be inside the annotated region.

- All annotations should represent the phone class.

- No incorrect or unrelated objects should be labeled as phones.

**Important:** Visual verification has been identified as the next task, but completion and execution of `visualize_dataset.py` have not yet been confirmed.

# 14. Current Completed Features

| Feature | Status |

|---|---|

| Python 3.12 setup | Completed |

| Virtual environment | Completed |

| Library installation | Completed |

| YOLO11 setup | Completed |

| YOLO person detection | Tested |

| YOLO cell phone detection | Tested |

| MediaPipe setup | Completed |

| Hand landmark detection | Tested |

| Combined YOLO + MediaPipe | Completed |

| Original image processing fix | Completed |

| Phone-in-hand test | Successful |

| Phone-on-desk test | Successful with workaround |

| Hand-to-phone proximity | Working |

| Annotated output image | Working |

| Phone detection dataset | Prepared |

| Dataset split verification | Completed |

| Dataset class-ID normalization | Completed |

| Dataset backup created | Completed |

| Dataset label validation | Completed |

| Latest train image/label count check | Confirmed |

| Latest validation image/label count check | Confirmed |

| Visual annotation verification | Pending |

| Custom YOLO training | Pending |

| Custom model evaluation | Pending |

| Live webcam testing | Pending hardware |

| Video processing | Not started |

| Temporal behaviour recognition | Not started |

| Phone usage timer | Not started |

| Database | Not started |

| Dashboard | Not started |

# 15. Known Limitations

## YOLO Classification

The pretrained YOLO11 nano model can misclassify phones depending on their position, angle, size, and image quality.

In our desk image, the phone was classified as a remote with 47.02% confidence.

The current remote workaround is only temporary.

A custom phone detection dataset may improve recognition.

## Hand Proximity

Hand proximity alone cannot determine whether a phone is being held.

For example:

- A hand may be near a phone without touching it.

- A phone may be lying on a desk.

- A hand may be holding another object.

- A phone may be held without visible hand landmarks covering the bounding box.

## Temporal Behaviour

The current system processes individual images.

It cannot yet reliably recognize:

- Reaching for a phone.

- Picking up a phone.

- Holding a phone over time.

- Typing on a phone.

- Putting a phone down.

These require video processing and temporal analysis.

## Webcam

No camera is currently available on the development PC.

Recorded videos and sample images will be used for development.

# 16. Planned Dataset Strategy

The project may use:

## FPI-DET

Potential dataset for phone detection.

Before relying on it, verify its availability, license, annotations, and class definitions.

## UCF101

Useful for general action recognition practice.

The existing 10-class subset is not sufficient by itself for phone-use recognition.

## Custom Behaviour Dataset

Eventually collect clips for:

1. Sitting normally.

2. Working on a laptop.

3. Reaching for phone.

4. Picking up phone.

5. Holding phone.

6. Looking at phone.

7. Typing on phone.

8. Talking on phone.

9. Putting phone down.

10. Walking away.

Begin with approximately 30–50 clips per class.

Include variation in:

- Lighting.

- Camera distance.

- Phone position.

- Hand position.

- Body posture.

- Recording sessions.

Split training, validation, and test data by recording session to avoid data leakage.

# 17. Next Development Roadmap

Continue from the current combined detection implementation.

Do not repeat completed installation or individual model setup.

## Step 1: Visually Verify the Custom Dataset

1. Create `visualize_dataset.py`.

2. Generate preview images for train, validation, and test.

3. Inspect bounding boxes.

4. Correct annotations only if actual visual problems are discovered.

5. Confirm that the dataset is suitable for training.

## Step 2: Validate the Dataset with Ultralytics

Perform a low-cost training sanity check or dataset loading check.

Confirm:

- `data.yaml` loads correctly.

- All three dataset splits are recognized.

- The model recognizes one class: `phone`.

- No corrupt images or invalid annotations cause problems.

## Step 3: Custom YOLO Training

Create:

```text

training/train_yolo.py

```

Use the pretrained model:

```text

yolo11n.pt

```

Fine-tune it using:

```text

dataset/phone_detection/data.yaml

```

Initial training configuration can start with:

- Model: YOLO11n.

- Image size: 640.

- Epochs: 50.

- Batch size: Adjust according to available hardware.

- Pretrained weights: Enabled.

- Dataset: Custom phone detection dataset.

Save the trained model under:

```text

models/phone_detector/

```

The final best checkpoint should be:

```text

models/phone_detector/best.pt

```

The exact training configuration may be adjusted based on available memory and training performance.

## Step 4: Evaluate the Custom Model

Evaluate:

- Precision.

- Recall.

- mAP50.

- mAP50-95.

- False positives.

- False negatives.

Use the test split for final evaluation after model selection.

## Step 5: Multiple Image Testing

Test the following:

- Phone held in hand.

- Phone lying on desk.

- Hand near phone.

- Hand away from phone.

- No phone visible.

Record YOLO classifications and confidence scores.

Investigate false positives before changing models.

## Step 6: Compare Pretrained and Custom YOLO

Compare:

```text

yolo11n.pt

```

Against:

```text

models/phone_detector/best.pt

```

Check whether the custom model improves phone detection, especially for phones lying on desks and unusual angles.

Do not assume the custom model is better without evaluating it.

## Step 7: Integrate the Custom Model

After successful evaluation, update the model-loading configuration in:

```text

combined_detection.py

```

Replace the pretrained model with the custom checkpoint.

Retest the existing phone-in-hand and phone-on-desk images.

Keep the original working model available for comparison.

The remote-class workaround should be reconsidered after evaluating the custom model. Do not automatically retain it if it causes false positives.

## Step 8: Recorded Video Processing

Build a new script:

```text

video_detection.py

```

The script should:

1. Load a recorded video.

2. Extract frames.

3. Run YOLO on each frame.

4. Run MediaPipe hand detection.

5. Track phone and hand positions.

6. Calculate hand-to-phone proximity.

7. Display detection results.

8. Save an annotated output video.

No webcam is required.

## Step 9: Temporal Behaviour Recognition

Track object positions across consecutive frames.

Recognize possible events:

- Hand approaching phone.

- Hand contacting phone.

- Phone moving with hand.

- Phone being held.

- Phone being put down.

- Hand moving away.

Use smoothing and confirmation windows to reduce false detections.

## Step 10: Behaviour State Machine

Planned states:

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

## Step 11: Usage Timer

Implement:

- Session start timestamp.

- Session end timestamp.

- Usage duration.

- Pickup count.

- Daily usage total.

- Average session duration.

## Step 12: VideoMAE

Explore fine-tuning:

```text

MCG-NJU/videomae-base

```

Use custom behaviour clips for temporal action recognition.

VideoMAE will complement YOLO; it will not replace the phone object detector.

## Step 13: Backend and Database

Develop:

- FastAPI backend.

- MongoDB session storage.

- Behaviour detection endpoints.

- Usage history endpoints.

## Step 14: Frontend Dashboard

Develop a React + Vite dashboard displaying:

- Current behaviour.

- Current phone-use status.

- Usage timer.

- Daily usage.

- Pickup count.

- Average session duration.

- Historical sessions.

- Charts and analytics.

## Step 15: Live Camera Integration

When webcam hardware becomes available:

- Test OpenCV camera capture.

- Connect the existing detection pipeline.

- Run real-time behaviour recognition.

- Test timer accuracy.

# 18. Instructions for Continuing Development

When continuing this project in a new ChatGPT conversation:

1. Read this README first.

2. Continue from the current completed implementation.

3. Do not reinstall Python or repeat completed setup.

4. Do not assume a webcam is available.

5. Do not jump directly to VideoMAE training or dashboard development.

6. Validate the existing detection pipeline before moving forward.

7. Provide complete code files when making substantial changes.

8. Give exact PowerShell commands.

9. Explain changes in beginner-friendly steps.

10. Avoid unnecessary changes to working components.

11. Give all related terminal commands together rather than scattering them across multiple messages.

12. Clearly explain which file to create, which existing code to replace, and where to paste it.

13. Wait for command output before proceeding to the next major step.

14. Do not repeat completed dataset normalization or backup operations without a specific reason.

15. Never assume a script has been created or executed unless its completion has been confirmed.

**Preferred guidance style:**

- Beginner-friendly explanations.

- Complete working code instead of disconnected snippets.

- Exact file names and locations.

- Exact PowerShell commands.

- Clear instructions about replacing existing code.

- One development step at a time.

- Verify the output before moving forward.

---

# 19. Current Working Scripts and Models

## Working Scripts

```text

test_yolo.py

detect_phone.py

test_hands.py

combined_detection.py

validate_dataset.py

camera_test.py

```

## Current Dataset

```text

dataset/phone_detection/

```

Dataset status:

- 852 training images and labels.

- 243 validation images and labels.

- 122 test images and labels.

- 1,217 total images.

- 1,217 matching label files.

- 1,462 phone annotations.

- One phone class intended in `data.yaml`.

- All annotations normalized to class ID `0`.

- Original labels preserved in `labels_backup_before_normalization/`.

- Structural validation completed.

- Latest PowerShell verification reconfirmed training and validation counts.

## Current Working Models

```text

yolo11n.pt

hand_landmarker.task

```

## Current Test Images

```text

sample_phone.jpg

phone_lying.jpg

```

## Current Output

```text

combined_detection_result.jpg

hand_detection_result.jpg

```

---

# 20. CURRENT PROJECT POSITION — NOVEMBER 19, 2026

**Last completed milestone: Custom phone detection dataset prepared, normalized, backed up, and structurally validated.**

The following work has been completed:

1. Python 3.12 environment configured.

2. Required computer vision libraries installed.

3. Pretrained YOLO11 model tested.

4. MediaPipe hand detection implemented.

5. Combined YOLO + MediaPipe detection implemented.

6. Phone-in-hand image tested successfully.

7. Phone-on-desk image tested using the temporary remote-class workaround.

8. Hand-to-phone proximity calculation implemented.

9. Custom phone dataset prepared with 1,217 images.

10. Original annotations backed up.

11. All 1,462 annotations normalized to class ID `0`.

12. Dataset structural validation completed.

13. Latest PowerShell command confirmed 852 training images and labels and 243 validation images and labels.

### NEXT IMMEDIATE TASK

**Create `visualize_dataset.py` and visually verify representative annotations from the training, validation, and test splits.**

After visual verification:

1. Validate the dataset with Ultralytics.

2. Train a custom YOLO11n model.

3. Evaluate precision, recall, mAP50, and mAP50-95.

4. Compare the custom model with pretrained YOLO.

5. Integrate the improved model into `combined_detection.py`.

The long-term objective remains a complete AI Personal Behaviour Monitoring System with automatic phone-use tracking, session history, and analytics.
21. Facial Behaviour Analysis — Live Webcam Implementation
Progress date: November 21, 2026

Current status: Facial landmark tracking and eye open/closed classification have been tested successfully. Blink counting is not yet reliable and remains under investigation.

This milestone adds a separate facial-analysis component to the AI Behaviour Monitor. Its purpose is to measure visible facial and eye-related signals during a live camera session. It complements the phone-and-hand detection pipeline; it does not replace YOLO phone detection or MediaPipe hand detection.
21.1 Development Context and What Was Done
The project initially focused on image-based face detection and facial landmark analysis. The face-detection helper module, face_detection.py, contains the shared model path, eye and iris landmark index groups, and reusable calculation/drawing functions.

A live facial-analysis script was then assembled to use those existing helpers rather than duplicate their implementation. The script is intended to read frames from an OpenCV camera source and run MediaPipe Face Landmarker in video mode.

A phone was explored as a camera source for the PC because the development PC did not have a built-in webcam. The current script uses OpenCV camera index 0; the actual camera connection method and stable device index should be confirmed during the next live run.

Earlier image testing showed that face detection can fail on blurry images or images containing multiple faces. A clearer image containing one face was subsequently detected successfully. The live script is configured for one face with num_faces=1.
21.2 Main Script and Dependencies
The live facial-analysis script is intended to be:
face_webcam_video_analysis.py
It imports the following shared items from face_detection.py:
- MODEL_PATH: location of the MediaPipe face landmark model.
- RIGHT_EYE and LEFT_EYE: landmark indices used to calculate each eye's Eye Aspect Ratio.
- RIGHT_IRIS and LEFT_IRIS: iris landmark index groups.
- landmark_to_pixel(): converts normalized landmark coordinates into image pixel coordinates.
- calculate_ear(): calculates Eye Aspect Ratio for an eye.
- calculate_iris_position(): estimates the iris position relative to the eye.
- estimate_head_pose(): estimates head pitch, yaw, and roll.
- draw_eye(): draws the eye landmark contour on the frame.
The script also uses:
Library	Role in this script
OpenCV (cv2)	Camera capture, frame conversion, drawing, display, and cleanup
MediaPipe Tasks	Face landmark detection and tracking in video mode
NumPy	Averaging iris coordinates and numerical operations
time	Session duration, closure timing, timestamps, and FPS
pathlib.Path	Resolving and checking the face model path


21.3 Live Processing Pipeline
For each camera frame, the script performs the following operations:
1. Checks that the face landmark model exists.
2. Opens the camera using cv2.VideoCapture(CAMERA_INDEX).
3. Requests a 1280 × 720 frame size.
4. Reads a frame and mirrors it horizontally for a mirror-like display.
5. Converts the frame from OpenCV BGR to RGB for MediaPipe.
6. Creates a MediaPipe image and supplies it to FaceLandmarker.detect_for_video().
7. Uses monotonically increasing millisecond timestamps because MediaPipe video mode requires ordered timestamps.
8. Processes the first detected face when face landmarks are available.
9. Draws a face bounding box and both eye contours.
10. Calculates eye, iris, gaze, and head-orientation measurements.
11. Displays live measurements and session counters on the camera frame.
12. Calculates FPS and exits when the user presses Q.
13. Releases the camera and closes OpenCV windows in a finally block.
14. Prints a final session report after the camera loop ends.
21.4 Eye Aspect Ratio (EAR)
The script calculates EAR separately for the right and left eyes, then averages the two values:
right_ear = calculate_ear(landmarks, RIGHT_EYE, width, height)
left_ear = calculate_ear(landmarks, LEFT_EYE, width, height)
average_ear = (right_ear + left_ear) / 2
EAR represents the relationship between the vertical and horizontal dimensions of the eye. When the eyelids close, the vertical eye opening becomes smaller and EAR generally decreases. When the eye is open, EAR generally increases.

The current script smooths the average EAR to reduce frame-to-frame fluctuations:
smoothed_ear = 0.65 * smoothed_ear + 0.35 * average_ear
The first measurement initializes the smoothed value. Subsequent measurements use 65% of the previous smoothed value and 35% of the newest average EAR.

Current initial threshold:
EAR_THRESHOLD = 0.20
The script classifies the eyes as closed when the smoothed EAR falls below this threshold. This is an initial heuristic, not a calibrated value for every camera, face, distance, or lighting condition.
21.5 Blink and Prolonged Eye-Closure Logic
Current timing configuration:
Setting	Current value	Purpose
MIN_BLINK_DURATION	0.08 seconds	Minimum closure duration considered for a blink event
MAX_BLINK_DURATION	0.60 seconds	Maximum closure duration counted as a normal blink
PROLONGED_CLOSURE_DURATION	0.60 seconds	Duration at which a closure is classified as prolonged
EAR_THRESHOLD	0.20	Threshold below which the smoothed EAR indicates closed eyes


The detection logic works as follows:
1. When smoothed EAR falls below the threshold, the script marks the eyes as closed and records the closure start time.
2. While the eyes remain below the threshold, it calculates the ongoing closure duration.
3. If the closure reaches 0.60 seconds, it changes the displayed state to PROLONGED CLOSURE and increments the prolonged-closure counter once for that closure.
4. When EAR rises above the threshold, the script treats the eyes as open and calculates the completed closure duration.
5. A closure lasting at least 0.08 seconds is included in total closure time.
6. A completed closure lasting from 0.08 through 0.60 seconds is counted as a normal blink, provided it was not already recorded as prolonged.
7. A closure longer than 0.60 seconds is not counted as a normal blink.
The script tracks the current closure separately so the on-screen total closure time includes an ongoing closure before the eyes reopen.
21.6 Iris and Approximate Gaze Estimation
The script calls calculate_iris_position() for both eyes and draws the iris landmark centres on the frame. It averages the two returned iris positions to estimate a rough horizontal gaze direction.

Current heuristic thresholds:
GAZE_LEFT_LIMIT = 0.35
GAZE_RIGHT_LIMIT = 0.65
The displayed categories are IMAGE LEFT, CENTER, IMAGE RIGHT, or UNKNOWN when the required iris values are unavailable.

These values are approximate image-relative gaze indicators. They are not calibrated eye-tracking measurements and should not be interpreted as proof of what a person is looking at.
21.7 Head Orientation Estimation
The script calls estimate_head_pose() and expects it to return pitch, yaw, and roll when a pose estimate is available.

Current thresholds:
HEAD_YAW_THRESHOLD = 20.0
HEAD_PITCH_THRESHOLD = 20.0
Yaw is used to classify a possible left or right head turn. Pitch is used to add a TILTED label when its absolute value exceeds the configured threshold. If pose estimation returns None, the display reports UNKNOWN.

The direction labels and signs must be checked with real camera movements because pose coordinate conventions can affect whether a positive yaw corresponds to a turn toward image-left or image-right. These thresholds are initial heuristics and have not been calibrated.
21.8 On-Screen Metrics and Session Report
The live display includes:
- Smoothed EAR value.
- Eye state (OPEN, CLOSED, or PROLONGED CLOSURE).
- Blink count.
- Blink rate per minute.
- Current eye-closure duration.
- Number of prolonged closures.
- Total eye-closure time, including the current closure.
- Approximate gaze direction.
- Estimated head direction.
- Head yaw and pitch values.
- Frames per second (FPS).
At shutdown, the script reports session duration, total frames, face-detection rate, total blinks, blink rate, prolonged closures, and accumulated eye-closure time.
21.9 Current Test Findings — November 21, 2026
The following observations have been reported during testing:
Test area	Observed result	Status
Clear single-face image detection	Face detection worked successfully on a clearer image after a blurry/multiple-face image failed	Working in the reported test
Face landmark processing	The live facial-analysis implementation uses MediaPipe face landmarks	Implemented; continue live validation
Eye open/closed classification	Open and closed eye states are being detected correctly	Working in the reported live test
Normal blink counting	Actual blinks are not being registered correctly by the counter	Needs debugging
Prolonged closure counting	Logic is implemented, but its live accuracy has not yet been confirmed	Not yet validated
Iris/gaze estimation	Implemented using iris landmark positions and heuristic thresholds	Not yet calibrated/validated
Head pose	Implemented through the shared estimate_head_pose() helper	Direction and thresholds need live validation
Session report and FPS	Implemented in the script	Output not yet confirmed in a completed session


Important distinction: Eye-state classification and blink-event detection are separate tasks. The reported result is that the eye state changes correctly between open and closed, but a blink is a short timed transition from open to closed and back to open. The current issue is therefore in the blink-event measurement/counting stage, not necessarily in face detection or eye landmark detection.
21.10 Blink-Counting Debugging Plan
Do not change face_detection.py or replace the working face/eye tracking code before inspecting the measurements. First collect actual EAR values from the current camera setup.

Temporarily print the smoothed EAR immediately after its calculation:
print(
    f"EAR: {smoothed_ear:.3f} | "
    f"State: {'CLOSED' if smoothed_ear < EAR_THRESHOLD else 'OPEN'}"
)
Record the approximate EAR range for:
- Eyes held naturally open.
- Eyes beginning to close.
- Eyes fully closed.
- Eyes reopening after a normal blink.
Then inspect whether the smoothed EAR crosses EAR_THRESHOLD during a normal blink and for how many frames. The threshold and smoothing factor should be adjusted only after observing these readings.

Potential causes to investigate, without assuming any one is the confirmed cause:
1. The fixed threshold of 0.20 may not fit the user's measured EAR range.
2. Smoothing may delay or reduce the brief EAR drop during a normal blink.
3. The 0.08-second minimum duration may reject a short closure.
4. The closure may not cross the threshold for enough consecutive frames to create a complete closed-to-open event.
5. Averaging both eyes may hide a closure if one eye's EAR behaves differently from the other.
6. A face-tracking gap resets the current closure state, so a closure interrupted by face loss is not completed and counted.
7. Camera frame rate and landmark jitter may affect the measured closure duration.
The next debugging step is to collect EAR readings during open eyes, normal blinking, and a one-second intentional eye closure. Do not tune thresholds by guessing.
21.11 Code Review Notes and Known Limitations
- BASE_DIR is defined in the script but is not currently used by the shown implementation.
- The script imports estimate_head_pose() from face_detection.py; that helper must exist and return values in the expected (pitch, yaw, roll) order.
- The gaze thresholds assume calculate_iris_position() returns comparable normalized horizontal positions for both eyes. This output convention should be confirmed in face_detection.py before interpreting gaze labels.
- The displayed IMAGE LEFT and IMAGE RIGHT labels refer to the mirrored camera frame, not necessarily the user's anatomical left and right.
- The blink counter increments only after a closure ends and its duration is checked. It will not increment at the instant the eyes close.
- A closure that reaches the prolonged threshold is recorded as prolonged once and is intentionally excluded from the normal blink count.
- When no face is detected, the script resets the current closure tracking and EAR smoothing state. It does not currently add an interrupted closure to the accumulated closure time.
- The current face bounding box is calculated from the minimum and maximum x/y values of all face landmarks. It is a landmark-extents rectangle, not a separately trained face detector's bounding box.
- Gaze direction and head orientation are heuristic estimates and require calibration/testing under different lighting, distances, and head positions.
- The camera index 0 is an assumption; if the phone-camera application exposes another device index, the configuration will need to be updated.
21.12 Current Facial Analysis Milestone and Next Actions
Completed in the current implementation:
1. Reused the existing face model and landmark helper functions.
2. Added a live OpenCV camera-processing loop using MediaPipe Face Landmarker video mode.
3. Added face and eye landmark drawing.
4. Added right-eye, left-eye, and average EAR calculation.
5. Added smoothed EAR and threshold-based eye open/closed classification.
6. Implemented timed blink and prolonged-closure counting logic.
7. Added iris landmark drawing and approximate gaze direction.
8. Added head pitch/yaw/roll estimation and heuristic direction labels.
9. Added live metrics, FPS, and a final session report.
10. Confirmed from the reported test that eye open/closed classification works, while normal blink counting still needs debugging.
Next immediate task: Print and record the smoothed EAR values during open eyes, normal blinks, and prolonged closure. Use those measurements to diagnose the blink counter before changing thresholds or modifying the shared face-detection module.

Do not mark blink detection, gaze calibration, or head-pose direction as fully validated until their live tests have been completed and the results recorded.