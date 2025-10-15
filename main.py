import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import face_recognition
from scipy.spatial import distance
import warnings

warnings.filterwarnings('ignore')


# =========================
# Utility Functions
# =========================
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar


# =========================
# Facial Landmark Visualization
# =========================
def highlight_facial_points(image_path):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(image_rgb, model='hog')
    for face_location in face_locations:
        landmarks = face_recognition.face_landmarks(image_rgb, [face_location])[0]
        for landmark_type, landmark_points in landmarks.items():
            for (x, y) in landmark_points:
                cv2.circle(image_rgb, (x, y), 3, (0, 255, 0), -1)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()


# =========================
# Drowsiness Detection Logic
# =========================
def process_image(frame):
    EYE_AR_THRESH = 0.25
    MOUTH_AR_THRESH = 0.6

    if frame is None:
        raise ValueError('Image is not found or unable to open')

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

    face_locations = face_recognition.face_locations(rgb_frame, model='hog')

    eye_flag = mouth_flag = False
    all_landmarks = []

    for face_location in face_locations:
        landmarks = face_recognition.face_landmarks(rgb_frame, [face_location])[0]
        all_landmarks.append(landmarks)

        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        mouth = np.array(landmarks['bottom_lip'])

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth)

        if ear < EYE_AR_THRESH:
            eye_flag = True
        if mar > MOUTH_AR_THRESH:
            mouth_flag = True

    return eye_flag, mouth_flag, all_landmarks


# =========================
# Draw Eyelid Outline & Distance
# =========================
def draw_eyelid_outline(image, eye_points, color=(0, 255, 0)):
    # Draw outline
    cv2.polylines(image, [eye_points], True, color, 2)

    # Get top and bottom points (for vertical distance)
    top_idx = 1  # upper eyelid
    bottom_idx = 5  # lower eyelid
    top_point = (eye_points[top_idx][0][0], eye_points[top_idx][0][1])
    bottom_point = (eye_points[bottom_idx][0][0], eye_points[bottom_idx][0][1])
    # Draw line showing eyelid distance
    cv2.line(image, top_point, bottom_point, (0, 0, 255), 2)

# =========================
# Real-time Webcam Loop
# =========================
video_cap = cv2.VideoCapture(0)  # webcam
count = 0
score = 0

while True:
    success, image = video_cap.read()
    if not success:
        break

    image = cv2.resize(image, (800, 500))
    count += 1

    n = 5  # process every 5 frames
    if count % n == 0:
        eye_flag, mouth_flag, all_landmarks = process_image(image)

        for landmarks in all_landmarks:
            left_eye = np.array(landmarks['left_eye'], np.int32).reshape((-1, 1, 2))
            right_eye = np.array(landmarks['right_eye'], np.int32).reshape((-1, 1, 2))

            # draw outlines + red distance line
            draw_eyelid_outline(image, left_eye)
            draw_eyelid_outline(image, right_eye)

        # ==== Update score ====
        if eye_flag or mouth_flag:
            score += 1
        else:
            score -= 1
            if score < 0:
                score = 0

    # ==== Display Score ====
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"Score: {score}", (10, image.shape[0] - 10), font, 1, (0, 255, 0), 2)

    # ==== Alert if Drowsy ====
    if score >= 5:
        cv2.putText(image, "Drowsy", (image.shape[1] - 150, 40), font, 1, (0, 0, 255), 2)

    cv2.imshow('Drowsiness Detection', image)

    # Press any key to exit
    if cv2.waitKey(1) & 0xFF != 255:
        break

video_cap.release()
cv2.destroyAllWindows()
