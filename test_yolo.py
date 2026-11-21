
from ultralytics import YOLO

print("Loading YOLO model...")

# Load pretrained YOLO11 nano model
model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully!")

# Check whether phone detection is supported
print("Phone class:", model.names[67])
print("Person class:", model.names[0])

print("\nYOLO setup completed!")
