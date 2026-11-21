
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from face_detection import (
    MODEL_PATH,
    RIGHT_EYE,
    LEFT_EYE,
    RIGHT_IRIS,
    LEFT_IRIS,
    landmark_to_pixel,
    calculate_ear,
    calculate_iris_position,
    draw_eye
)


def process_webcam(camera_index=0):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Face model not found: {MODEL_PATH}"
        )

    # Open webcam
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}"
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam connected successfully!")
    print("Starting live facial landmark detection...")
    print("Press Q to exit.")

    # MediaPipe Face Landmarker
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    fps = 0.0
    previous_time = time.perf_counter()

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:

                success, frame = camera.read()

                if not success:
                    print("Could not read webcam frame.")
                    break

                # Mirror webcam image
                frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # VIDEO mode requires increasing timestamps
                timestamp_ms = time.monotonic_ns() // 1_000_000

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                if result.face_landmarks:

                    landmarks = result.face_landmarks[0]

                    # Calculate face bounding box
                    x_values = [point.x for point in landmarks]
                    y_values = [point.y for point in landmarks]

                    x1 = max(0, int(min(x_values) * width))
                    y1 = max(0, int(min(y_values) * height))
                    x2 = min(width - 1, int(max(x_values) * width))
                    y2 = min(height - 1, int(max(y_values) * height))

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Draw facial landmarks
                    for point in landmarks:

                        px, py = landmark_to_pixel(
                            point,
                            width,
                            height
                        )

                        cv2.circle(
                            frame,
                            (px, py),
                            1,
                            (0, 255, 0),
                            -1
                        )

                    # Highlight eye contours
                    draw_eye(
                        frame,
                        landmarks,
                        RIGHT_EYE,
                        width,
                        height
                    )

                    draw_eye(
                        frame,
                        landmarks,
                        LEFT_EYE,
                        width,
                        height
                    )

                    # Calculate EAR
                    right_ear = calculate_ear(
                        landmarks,
                        RIGHT_EYE,
                        width,
                        height
                    )

                    left_ear = calculate_ear(
                        landmarks,
                        LEFT_EYE,
                        width,
                        height
                    )

                    average_ear = (
                        right_ear + left_ear
                    ) / 2

                    # Estimate iris positions
                    right_iris = calculate_iris_position(
                        landmarks,
                        RIGHT_IRIS,
                        RIGHT_EYE,
                        width,
                        height
                    )

                    left_iris = calculate_iris_position(
                        landmarks,
                        LEFT_IRIS,
                        LEFT_EYE,
                        width,
                        height
                    )

                    # Draw iris centres
                    for iris_indices in [
                        RIGHT_IRIS,
                        LEFT_IRIS
                    ]:

                        iris_points = [
                            landmark_to_pixel(
                                landmarks[index],
                                width,
                                height
                            )
                            for index in iris_indices
                        ]

                        centre = (
                            int(np.mean([
                                p[0] for p in iris_points
                            ])),
                            int(np.mean([
                                p[1] for p in iris_points
                            ]))
                        )

                        cv2.circle(
                            frame,
                            centre,
                            5,
                            (255, 0, 255),
                            -1
                        )

                    # Display facial information
                    cv2.putText(
                        frame,
                        "FACE DETECTED",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Landmarks: {len(landmarks)}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Average EAR: {average_ear:.3f}",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    if (
                        right_iris is not None
                        and left_iris is not None
                    ):

                        cv2.putText(
                            frame,
                            f"Iris R/L: {right_iris:.2f} / {left_iris:.2f}",
                            (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 0, 255),
                            2
                        )

                else:

                    cv2.putText(
                        frame,
                        "NO FACE DETECTED",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                # Calculate FPS
                current_time = time.perf_counter()

                elapsed = current_time - previous_time
                previous_time = current_time

                if elapsed > 0:
                    current_fps = 1.0 / elapsed

                    if fps == 0:
                        fps = current_fps
                    else:
                        fps = 0.9 * fps + 0.1 * current_fps

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (width - 170, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.imshow(
                    "AI Behaviour Monitor - Live Face Analysis",
                    frame
                )

                # Exit when Q is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

        print("\nWebcam session ended.")
        print("Camera released successfully.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Live facial landmark analysis"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index"
    )

    args = parser.parse_args()

    process_webcam(args.camera)
