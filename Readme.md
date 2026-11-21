py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install opencv-python ultralytics mediapipe numpy pandas matplotlib


python -c "import cv2, ultralytics, mediapipe, numpy, pandas, matplotlib; print('All libraries installed successfully!')"



python camera_test.py
python test_yolo.py



python -c "import mediapipe as mp; print('MediaPipe:', mp.__version__); print('Solutions available:', hasattr(mp, 'solutions'))"


yolo detect train data="D:\AI-Behaviour-Monitor\dataset\phone_detection\data.yaml" model=yolo11n.pt epochs=1 imgsz=64 batch=1 device=cpu plots=False


yolo detect train data="D:\AI-Behaviour-Monitor\dataset\phone_detection\data.yaml" model=yolo11n.pt epochs=10 imgsz=640 batch=8 device=cpu plots=True name=phone_v1





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