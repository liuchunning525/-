import os
import cv2
import time
import joblib
import numpy as np
import mediapipe as mp
import pandas as pd
from collections import Counter, deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

NUM_HANDS = 1
VIDEO_SOURCE = 1
SMOOTH_WINDOW = 10
TARGET_LANDMARK_INDEX = 8
INDEX_TIP_HISTORY = deque(maxlen=5)

POSE_UPDATE_INTERVAL = 2
DISPLAY_LANDMARKS = [0, 4, 5, 8, 9, 12, 13, 17]

def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_camera_pose(project_root: str):
    calib_path = os.path.join(project_root, "camera", "camera_calibration.npz")
    pose_path = os.path.join(project_root, "camera", "camera_pose.npz")

    if not os.path.exists(calib_path):
        print(f"[ERROR] Camera calibration not found: {calib_path}")
        return None, None, None, None

    if not os.path.exists(pose_path):
        print(f"[ERROR] Camera pose not found: {pose_path}")
        return None, None, None, None

    calib = np.load(calib_path, allow_pickle=True)
    pose = np.load(pose_path, allow_pickle=True)

    camera_matrix = calib["camera_matrix"]
    dist_coeffs = calib["dist_coeffs"]
    R_wc = pose["R"]
    t_wc = pose["tvec"]

    return camera_matrix, dist_coeffs, R_wc, t_wc

def create_detector(model_path: str, num_hands: int = 1):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands
    )
    return vision.HandLandmarker.create_from_options(options)

def get_recommendation(label: str) -> str:
    recommendations = {
        "palm": "Larger mouse, stronger palm support",
        "claw": "Medium size, higher back hump",
        "fingertip": "Smaller and lighter mouse"
    }
    return recommendations.get(label, "No recommendation available")

def extract_features_from_result(detection_result):
    if not detection_result.hand_landmarks:
        return None

    hand_landmarks = detection_result.hand_landmarks[0]
    features = []

    for lm in hand_landmarks:
        features.extend([lm.x, lm.y, lm.z])

    columns = []
    for i in range(21):
        columns.extend([f"x{i}", f"y{i}", f"z{i}"])

    return pd.DataFrame([features], columns=columns)

def draw_landmarks_on_frame(frame_bgr, detection_result):
    annotated = frame_bgr.copy()

    if not detection_result.hand_landmarks:
        return annotated

    height, width, _ = annotated.shape

    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    for hand_landmarks in detection_result.hand_landmarks:
        points = []

        for lm in hand_landmarks:
            x = int(lm.x * width)
            y = int(lm.y * height)
            points.append((x, y))

        for point in points:
            cv2.circle(annotated, point, 4, (0, 255, 0), -1)

        for start, end in hand_connections:
            cv2.line(annotated, points[start], points[end], (255, 0, 0), 2)

    return annotated

def get_smoothed_label(history):
    if not history:
        return None
    return Counter(history).most_common(1)[0][0]

def estimate_hand_pose(hand_landmarks, hand_world_landmarks, width, height, camera_matrix, dist_coeffs):
    image_points = []
    object_points = []

    for lm2d, lm3d in zip(hand_landmarks, hand_world_landmarks):
        image_points.append([lm2d.x * width, lm2d.y * height])
        object_points.append([lm3d.x * 100.0, lm3d.y * 100.0, lm3d.z * 100.0])

    image_points = np.array(image_points, dtype=np.float32)
    object_points = np.array(object_points, dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_SQPNP
    )

    if not success:
        return None, None

    R_hc, _ = cv2.Rodrigues(rvec)
    return R_hc, tvec

def hand_point_to_world(point_hand_cm, R_hc, t_hc, R_wc, t_wc):
    point_hand_cm = np.asarray(point_hand_cm, dtype=np.float64).reshape(3, 1)
    point_cam = R_hc @ point_hand_cm + t_hc
    point_world = R_wc.T @ (point_cam - t_wc)
    return point_world

def main():
    project_root = get_project_root()
    detector_model_path = os.path.join(project_root, "hand_landmarker.task")
    classifier_path = os.path.join(project_root, "models", "grip_model.pkl")

    if not os.path.exists(detector_model_path):
        print(f"[ERROR] Hand landmarker model not found: {detector_model_path}")
        return

    if not os.path.exists(classifier_path):
        print(f"[ERROR] Trained classifier not found: {classifier_path}")
        print("Run train_classifier.py first.")
        return

    camera_matrix, dist_coeffs, R_wc, t_wc = load_camera_pose(project_root)
    if camera_matrix is None:
        return

    detector = create_detector(detector_model_path, NUM_HANDS)
    clf = joblib.load(classifier_path)

    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {VIDEO_SOURCE}")
        return

    prediction_history = deque(maxlen=SMOOTH_WINDOW)
    frame_count = 0
    last_R_hc = None
    last_t_hc = None

    prev_time = time.time()
    print("Press 'q' to quit.")

    landmark_names = {
        0: "wrist",
        1: "thumb_cmc",
        2: "thumb_mcp",
        3: "thumb_ip",
        4: "thumb_tip",
        5: "index_mcp",
        6: "index_pip",
        7: "index_dip",
        8: "index_tip",
        9: "middle_mcp",
        10: "middle_pip",
        11: "middle_dip",
        12: "middle_tip",
        13: "ring_mcp",
        14: "ring_pip",
        15: "ring_dip",
        16: "ring_tip",
        17: "pinky_mcp",
        18: "pinky_pip",
        19: "pinky_dip",
        20: "pinky_tip",
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        frame_count += 1

        if VIDEO_SOURCE == 0:
            frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = detector.detect(mp_image)
        annotated_frame = draw_landmarks_on_frame(frame, detection_result)

        landmark_world_coords = {}

        if detection_result.hand_landmarks and detection_result.hand_world_landmarks:
            height, width, _ = frame.shape

            hand_landmarks = detection_result.hand_landmarks[0]
            hand_world_landmarks = detection_result.hand_world_landmarks[0]

            if frame_count % POSE_UPDATE_INTERVAL == 0 or last_R_hc is None or last_t_hc is None:
                R_hc, t_hc = estimate_hand_pose(
                    hand_landmarks,
                    hand_world_landmarks,
                    width,
                    height,
                    camera_matrix,
                    dist_coeffs
                )

                if R_hc is not None:
                    last_R_hc = R_hc
                    last_t_hc = t_hc

            R_hc = last_R_hc
            t_hc = last_t_hc

            if R_hc is not None:
                for i, (lm2d, lm3d) in enumerate(zip(hand_landmarks, hand_world_landmarks)):
                    pixel_x = lm2d.x * width
                    pixel_y = lm2d.y * height

                    hand_point_cm = np.array(
                        [lm3d.x * 100.0, lm3d.y * 100.0, lm3d.z * 100.0],
                        dtype=np.float64
                    )

                    world_point = hand_point_to_world(
                        hand_point_cm,
                        R_hc,
                        t_hc,
                        R_wc,
                        t_wc
                    )

                    real_x = float(world_point[0, 0])
                    real_y = float(world_point[1, 0])
                    real_z = float(world_point[2, 0])

                    landmark_world_coords[i] = {
                        "name": landmark_names.get(i, f"lm_{i}"),
                        "pixel_x": pixel_x,
                        "pixel_y": pixel_y,
                        "real_x": real_x,
                        "real_y": real_y,
                        "real_z": real_z,
                    }

                for idx in DISPLAY_LANDMARKS:
                    if idx not in landmark_world_coords:
                        continue
                    info = landmark_world_coords[idx]
                    px = int(info["pixel_x"])
                    py = int(info["pixel_y"])
                    real_x = info["real_x"]
                    real_y = info["real_y"]
                    real_z = info["real_z"]

                    if idx == TARGET_LANDMARK_INDEX:
                        INDEX_TIP_HISTORY.append([real_x, real_y, real_z])
                        avg_point = np.mean(np.array(INDEX_TIP_HISTORY), axis=0)
                        display_x = float(avg_point[0])
                        display_y = float(avg_point[1])
                        display_z = float(avg_point[2])
                    else:
                        display_x = real_x
                        display_y = real_y
                        display_z = real_z

                    cv2.putText(
                        annotated_frame,
                        f"{idx}:({display_x:.1f},{display_y:.1f},{display_z:.1f})",
                        (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 255, 255),
                        1
                    )

        features = extract_features_from_result(detection_result)

        display_label = None
        confidence_text = ""

        if features is not None:
            pred_label = clf.predict(features)[0]
            prediction_history.append(pred_label)
            display_label = get_smoothed_label(prediction_history)

            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(features)[0]
                confidence = float(np.max(probs))
                confidence_text = f"{confidence:.2f}"

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 1e-6)
        prev_time = current_time

        if display_label is not None:
            recommendation = get_recommendation(display_label)

            cv2.putText(
                annotated_frame,
                f"Grip: {display_label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            if confidence_text:
                cv2.putText(
                    annotated_frame,
                    f"Confidence: {confidence_text}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            cv2.putText(
                annotated_frame,
                f"Recommend: {recommendation}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        else:
            cv2.putText(
                annotated_frame,
                "No hand detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.imshow("Mouse Grip Video Prediction", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()