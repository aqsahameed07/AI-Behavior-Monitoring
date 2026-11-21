
import cv2

# Open laptop webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Error: Could not access webcam!")
    exit()

print("Webcam connected successfully!")
print("Press Q to close the camera.")

while True:
    success, frame = camera.read()

    if not success:
        print("Failed to capture frame.")
        break

    # Display camera feed
    cv2.imshow("AI Behaviour Monitor - Webcam Test", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

print("Webcam test completed.")
