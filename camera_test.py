
import cv2

print("Searching for available cameras...\n")

for backend_name, backend in [
    ("DirectShow", cv2.CAP_DSHOW),
    ("Media Foundation", cv2.CAP_MSMF),
]:

    print(f"\nTesting {backend_name}")

    for index in range(6):

        cap = cv2.VideoCapture(index, backend)

        if not cap.isOpened():
            cap.release()
            continue

        ret, frame = cap.read()

        if ret and frame is not None:
            print(
                f"Camera {index}: WORKING | "
                f"Resolution: {frame.shape[1]}x{frame.shape[0]}"
            )

            cv2.imshow(
                f"{backend_name} - Camera {index}",
                frame
            )

            cv2.waitKey(1500)
            cv2.destroyAllWindows()

        else:
            print(f"Camera {index}: Opened but NO FRAME")

        cap.release()

print("\nCamera scan completed.")
