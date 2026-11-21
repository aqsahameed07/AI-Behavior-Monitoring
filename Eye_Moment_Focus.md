AI Behaviour Monitor — Fusion Monitor Documentation
File: focus_fusion_monitor.py
Date: November 21, 2026
Status: Code delivered; awaiting first-run test results

1. What This Module Does
The fusion monitor combines four independent signal sources into a single behavior classification and honest focus score. It replaces the earlier face-only face_webcam_video_analysis.py (which remains untouched for comparison).

Signal source	What it provides	Library / Model
Face landmarks + blendshapes	EAR, gaze, head pose, smile, jaw-open (yawn)	MediaPipe Face Landmarker (face_landmarker.task)
Hand landmarks	Hand position, proximity to objects	MediaPipe Hand Landmarker (hand_landmarker.task)
Object detection	Phone, bottle, cup, remote boxes	YOLO26n (yolo26n.pt)
Input activity	Keyboard/mouse event timestamps only	pynput
YOLO runs only when a hand is present. A bottle on the desk with no hand near it is not a distraction, so we don't waste CPU on it. This keeps FPS reasonable on CPU-only PyTorch.

2. Why YOLO26n (Not YOLO11n)
Reason	Detail
CPU inference speed	YOLO26 is up to 43% faster on CPU than YOLO11
No NMS	End-to-end inference, lower latency, simpler code
Same API	YOLO("yolo26n.pt") — drop-in replacement
Same COCO classes	cell phone (67), bottle (39), cup (41), remote (65)
Your ultralytics version (8.4.215) already supports YOLO26. No reinstall needed.

3. Behaviour Taxonomy (What the System Distinguishes)
The classification tree runs in priority order — first match wins. Each label comes with a confidence value and a focus percentage.

Priority	Label	Trigger condition	Focus %
1	DROWSY / ZONED OUT	Eyes closed > 0.6s continuously	5%
2	PHONE DISTRACTION	Cell phone detected + hand within 60px	10%
3	YARNING / FATIGUED	≥2 yawns in last 60s	15%
4	RESTLESS	Gaze variance > 0.060 AND head variance > 1.5	20%
5a	WATCHING (passive)	Gaze var low + head var low + no typing + smiling	12%
5b	ZONED OUT / STARING	Gaze var low + head var low + no typing + not smiling	10%
6	FIDGETING	Hand near distraction object + no typing	25%
7	DISTRACTED	Gaze not center OR head turned	20%
8	WORKING	Typing + gaze center + gaze variance above floor	88%
9a	READING + NOTES	Medium gaze variance + occasional typing	82%
9b	READING	Medium gaze variance + no typing	68%
10	LIGHT ACTIVITY	Mouse active + gaze center, nothing else matches	50%
—	NO FACE	Face not detected	frozen
—	CALIBRATING	First 3 seconds at startup	0%
4. Rolling Behavior Window
Instead of classifying each frame in isolation (which flickers), we keep a 10-second rolling buffer of per-frame signals and compute statistics over it.

python
class BehaviorWindow:
    # deque of (timestamp, {signal: value})
    # stats(key) returns mean, std, count
Computed every frame once the window has ≥15 samples:

gaze_var_10s — standard deviation of iris-x offset

head_var_10s — combined std of yaw + pitch + roll

These variance values (not raw instantaneous values) are what drive the classification. This is what makes the distinction between "reading" (moderate saccades) and "staring" (near-zero variance) actually work.

5. Distraction Signals You Confirmed
You explicitly named these three as your distraction vocabulary:

Your wording	Implemented as
"Yawning"	Blendshape jawOpen > 0.55 for >1.0s, counted in 60s window
"Looking here and there frequently"	gaze_var > 0.060 AND head_var > 1.5 → RESTLESS
"Frequent mobile usage"	YOLO cell phone + hand within 60px → PHONE DISTRACTION
Plus the pattern from earlier discussions:

Your insight	Implemented as
"Staring + smiling + no typing = watching Netflix"	WATCHING (passive) — requires smile blendshape > 0.30 within last 10s
"Playing with bottle = not focused"	FIDGETING — hand near non-phone distraction object + no typing
"Researching/reading looks like thinking + typing"	READING + NOTES — medium gaze variance + keyboard activity
6. Calibration
EAR calibration (automatic, 3 seconds, built-in)
On startup, the script measures your natural open-eye EAR for 3 seconds, then derives:

text
closed_estimate = open_estimate × 0.40
EAR_THRESHOLD   = closed_estimate + 0.30 × (open_estimate − closed_estimate)
Printed to terminal once: >>> EAR calibrated: open~X, closed~Y, threshold=Z <<<

Behavior variance calibration (NOT yet enabled)
CALIBRATION_ENABLED = False by default. The thresholds GAZE_VAR_LOW = 0.008, GAZE_VAR_HIGH = 0.060, HEAD_VAR_LOW = 1.5, HEAD_VAR_HIGH = 6.0 are educated guesses.

Next step after first run: collect 10–15 minutes of real data, inspect the CSV, and tune these values against your actual variance ranges during work vs. reading vs. watching. A guided 90-second calibration mode (work 30s / read 30s / watch 30s → auto-set thresholds) can be added once we see your real numbers.

7. Privacy Boundary
Signal	Captured	Not captured
Keyboard	Timestamp of keypress	Which key, what text
Mouse	Movement timestamps + position deltas	Screen content
Camera	Face/hand landmarks, object boxes	Raw video is not saved
CSV log	Numeric signals + labels per frame	Images or video
All CSV logs are saved locally in reports/. Nothing is sent over the network.

8. Requirements
New dependency (only one)
powershell
pip install pynput
Everything else is already installed per your project documentation.

Files required in D:\AI-Behaviour-Monitor\
File	Purpose	Status
face_landmarker.task	MediaPipe face model	Already present
hand_landmarker.task	MediaPipe hand model	Already present
yolo26n.pt	YOLO26 nano weights	Will auto-download on first YOLO("yolo26n.pt") call
venv\	Virtual environment	Active
Python packages used
text
opencv-python
mediapipe
ultralytics      (>= 8.4.0 for YOLO26)
numpy
pynput           ← NEW
Removed dependencies from this module
Removed	Reason
face_detection.py imports	Inlined EAR, gaze, and head pose helpers so the file is self-contained. You can still import from face_detection.py in other scripts.
estimate_head_pose() from helper	Replaced with a simpler nose-offset heuristic (yaw/pitch only, no roll). If full solvePnP head pose is needed later, we can wire the original helper back in.
9. Known Limitations (Version 1)
Limitation	Impact	Fix path
Head pose is nose-offset heuristic	Yaw/pitch signs may need flipping; roll = 0	Wire in estimate_head_pose() from face_detection.py if needed
Gaze variance thresholds are guesses	Misclassification possible	Tune with CSV data after 10–15 min run
No guided calibration for behavior variance	First runs may be noisy	Add 90s calibration once thresholds are known
YOLO26n only fires when a hand is detected	A phone on desk with no hand is invisible	Acceptable tradeoff — intent was to reduce CPU
Smile blendshape may fire on natural resting face for some people	False "WATCHING" classifications possible	Tune SMILE_THRESHOLD (currently 0.30) after real runs
Behavior window clears on face-loss	Brief tracking gaps reset variance	Current behavior; may add a "gap tolerance" counter later
No mouse-smoothness metric	Can't distinguish precise work-mouse from idle fidget-mouse	Planned addition
No fixation-duration metric	Can't compute "longest stare" directly	Planned addition
10. What We Decided to Add Later (Backlog)
These were discussed and agreed as valuable, but not in Version 1:

Addition	Why	Priority
Guided behavior calibration (90s: work/read/watch)	Removes guesswork from variance thresholds	High
Mouse movement smoothness (intermittency, correction frequency)	Distinguishes precise work-mouse from fidget	High
Fixation duration metric	Direct "how long did they stare" number	High
Saccade frequency (per second)	Sharper read-vs-stare distinction than variance alone	Medium
Combined yawn + blink-duration fatigue logic	Reduces false yawning (single yawn ≠ fatigue)	Medium
Distraction recovery time	Tracks how fast user returns to focus after a distraction	Medium
Full head pose (solvePnP / estimate_head_pose)	Correct yaw/pitch/roll, no sign guessing	Low
Shoulder / upper-body posture	Tiebreaker for "leaning in to focus" vs "leaning back"	Low
VideoMAE action recognition	Only if rule-based fusion proves insufficient	Low / research
11. First-Run Checklist
Before declaring Version 1 successful, run through this:

#	Action	Expected result
1	pip install pynput	Install succeeds
2	python focus_fusion_monitor.py	YOLO26n downloads, all models load, camera opens
3	Hold eyes open 3 seconds	Terminal prints >>> EAR calibrated: open~X, closed~Y, threshold=Z <<<
4	Blink 5 times	Eye state toggles OPEN → CLOSED → OPEN; CSV shows raw_ear dips
5	Yawn once for 1.5s	On-screen Yawn count increments after ~1s
6	Yawn 2 more times	Label flips to YARNING / FATIGUED
7	Stare blankly at screen 10s	Label flips to ZONED OUT / STARING, focus < 20%
8	Stare + smile at screen 10s	Label flips to WATCHING (passive)
9	Type in editor 15s	Label flips to WORKING, focus > 80%
10	Play with bottle 10s	Label flips to FIDGETING
11	Hold phone near face 10s	Label flips to PHONE DISTRACTION
12	Look rapidly around room 10s	Label flips to RESTLESS
13	Quit with Q	Session report prints; CSV path shown
14	Open CSV in Excel	All columns populated; no empty label rows
12. What to Report Back After First Run
Paste the following so the next iteration can be tuned:

The >>> EAR calibrated terminal line (open / closed / threshold values)

The session report printed at quit

Whether any of the checklist rows 4–12 misbehaved (and how)

Your actual gaze variance ranges from the CSV — take a 2-minute sample of "clearly working" and a 2-minute sample of "clearly watching," report the mean gaze_var_10s in each. Those two numbers will let us set GAZE_VAR_LOW and GAZE_VAR_HIGH correctly instead of the current guesses.

Any traceback if a step crashed

Once that data is in, the next iteration adds:

Guided 90s calibration

Mouse smoothness metric

Fixation duration

Tuned thresholds that match your face and your camera

End of documentation.
File to run: focus_fusion_monitor.py
New dependency: pynput
Logs: reports/focus_fusion_YYYYMMDD_HHMMSS.csv

