
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker" / "face_landmarker.task"

# MediaPipe facial landmark indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_IRIS = [473, 474, 475, 476, 477]


def landmark_to_pixel(landmark, width, height):
    x = int(landmark.x * width)
    y = int(landmark.y * height)

    return (
        max(0, min(width - 1, x)),
        max(0, min(height - 1, y))
    )


def calculate_distance(point1, point2):
    return np.linalg.norm(
        np.array(point1, dtype=float) -
        np.array(point2, dtype=float)
    )


def calculate_ear(landmarks, eye_indices, width, height):
    """
    Calculate Eye Aspect Ratio.

    Lower EAR can indicate that the eye is closed.
    It does not independently prove a blink.
    """

    points = [
        landmark_to_pixel(landmarks[index], width, height)
        for index in eye_indices
    ]

    vertical_1 = calculate_distance(points[1], points[5])
    vertical_2 = calculate_distance(points[2], points[4])
    horizontal = calculate_distance(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_iris_position(landmarks, iris_indices, eye_indices,
                            width, height):
    """
    Estimate iris position relative to the eye corners.

    This is an approximate image-relative position,
    not calibrated gaze tracking.
    """

    iris_points = [
        landmark_to_pixel(landmarks[index], width, height)
        for index in iris_indices
    ]

    iris_x = np.mean([point[0] for point in iris_points])

    corner_1 = landmark_to_pixel(
        landmarks[eye_indices[0]], width, height
    )
    corner_2 = landmark_to_pixel(
        landmarks[eye_indices[3]], width, height
    )

    left_x = min(corner_1[0], corner_2[0])
    right_x = max(corner_1[0], corner_2[0])

    eye_width = right_x - left_x

    if eye_width == 0:
        return None

    return float((iris_x - left_x) / eye_width)


def estimate_head_pose(landmarks, width, height):
    """
    Rough head orientation using solvePnP.

    This is an approximation, not calibrated 3D head tracking.
    """

    landmark_ids = [1, 152, 33, 263, 61, 291]

    image_points = np.array(
        [
            landmark_to_pixel(landmarks[index], width, height)
            for index in landmark_ids
        ],
        dtype=np.float64
    )

    # Approximate generic 3D face reference points.
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose
        (0.0, -63.6, -12.5),      # Chin
        (-43.3, 32.7, -26.0),     # Left eye outer corner
        (43.3, 32.7, -26.0),      # Right eye outer corner
        (-28.9, -28.9, -24.1),    # Left mouth corner
        (28.9, -28.9, -24.1)      # Right mouth corner
    ], dtype=np.float64)

    focal_length = float(width)

    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    angles = cv2.RQDecomp3x3(rotation_matrix)[0]

    return tuple(float(angle) for angle in angles)


def draw_eye(frame, landmarks, indices, width, height):
    points = np.array([
        landmark_to_pixel(landmarks[index], width, height)
        for index in indices
    ], dtype=np.int32)

    cv2.polylines(
        frame,
        [points],
        True,
        (0, 255, 255),
        2
    )


def process_image(image_path, output_path):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Face model not found: {MODEL_PATH}\n"
            "Download face_landmarker.task first."
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    height, width = image.shape[:2]

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        print("No face detected in this image.")

    else:
        landmarks = result.face_landmarks[0]

        # Face bounding box
        x_values = [point.x for point in landmarks]
        y_values = [point.y for point in landmarks]

        x1 = max(0, int(min(x_values) * width))
        y1 = max(0, int(min(y_values) * height))
        x2 = min(width - 1, int(max(x_values) * width))
        y2 = min(height - 1, int(max(y_values) * height))

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw eye contours
        draw_eye(image, landmarks, RIGHT_EYE, width, height)
        draw_eye(image, landmarks, LEFT_EYE, width, height)

        # Calculate eye aspect ratios
        right_ear = calculate_ear(
            landmarks, RIGHT_EYE, width, height
        )

        left_ear = calculate_ear(
            landmarks, LEFT_EYE, width, height
        )

        average_ear = (right_ear + left_ear) / 2

        # A starting threshold only. Calibrate for real video.
        eye_state = (
            "CLOSED-LIKE"
            if average_ear < 0.20
            else "OPEN-LIKE"
        )

        # Draw iris centres
        for iris_indices in [RIGHT_IRIS, LEFT_IRIS]:
            iris_points = [
                landmark_to_pixel(
                    landmarks[index], width, height
                )
                for index in iris_indices
            ]

            centre = (
                int(np.mean([point[0] for point in iris_points])),
                int(np.mean([point[1] for point in iris_points]))
            )

            cv2.circle(
                image,
                centre,
                5,
                (255, 0, 255),
                -1
            )

        # Approximate iris positions
        right_iris = calculate_iris_position(
            landmarks, RIGHT_IRIS, RIGHT_EYE, width, height
        )

        left_iris = calculate_iris_position(
            landmarks, LEFT_IRIS, LEFT_EYE, width, height
        )

        # Estimate head orientation
        head_pose = estimate_head_pose(
            landmarks, width, height
        )

        # Display information
        cv2.putText(
            image,
            f"Face detected | Landmarks: {len(landmarks)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"EAR: {average_ear:.3f} | Eyes: {eye_state}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        if right_iris is not None and left_iris is not None:
            cv2.putText(
                image,
                f"Iris position R/L: {right_iris:.2f} / {left_iris:.2f}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2
            )

        if head_pose is not None:
            px, py, pz = head_pose

            cv2.putText(
                image,
                f"Head rotation X/Y/Z: {px:.1f} / {py:.1f} / {pz:.1f}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

        print("\nFACE ANALYSIS")
        print("-------------------------")
        print(f"Landmarks detected: {len(landmarks)}")
        print(f"Right EAR: {right_ear:.3f}")
        print(f"Left EAR: {left_ear:.3f}")
        print(f"Average EAR: {average_ear:.3f}")
        print(f"Eye state: {eye_state}")
        print(f"Right iris position: {right_iris}")
        print(f"Left iris position: {left_iris}")
        print(f"Head rotation X/Y/Z: {head_pose}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), image)

    print(f"\nOutput saved to: {output_path.resolve()}")

    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Face and eye landmark detection"
    )

    parser.add_argument(
        "--image",
        default="sample_phone.jpg",
        help="Input image path"
    )

    parser.add_argument(
        "--output",
        default="runs/face_detection/face_output.jpg",
        help="Output image path"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.is_absolute():
        image_path = BASE_DIR / image_path

    output_path = Path(args.output)

    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    process_image(image_path, output_path)
