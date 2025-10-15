import cv2
import numpy as np
from scipy.spatial import distance
import face_recognition
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
# Drowsiness Detection Logic
# =========================
def process_image(frame):
    EYE_AR_THRESH = 0.21  # 👈 tuned threshold
    MOUTH_AR_THRESH = 0.6

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

    face_locations = face_recognition.face_locations(rgb_frame, model='hog')

    left_eye_flag = right_eye_flag = mouth_flag = False
    left_ear = right_ear = 0.0
    all_landmarks = []

    for face_location in face_locations:
        landmarks = face_recognition.face_landmarks(rgb_frame, [face_location])[0]
        all_landmarks.append(landmarks)

        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        mouth = np.array(landmarks['bottom_lip'])

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        mar = mouth_aspect_ratio(mouth)

        if left_ear < EYE_AR_THRESH:
            left_eye_flag = True
        if right_ear < EYE_AR_THRESH:
            right_eye_flag = True
        if mar > MOUTH_AR_THRESH:
            mouth_flag = True

    return left_eye_flag, right_eye_flag, mouth_flag, all_landmarks, left_ear, right_ear

# =========================
# Draw Eyelid Outline
# =========================
def draw_eyelid_outline(image, eye_points, color=(0, 255, 0)):
    cv2.polylines(image, [eye_points], True, color, 2)

# =========================
# Real-time Webcam Loop
# =========================
video_cap = cv2.VideoCapture(0)
count = 0
score = 0

# 👇 Keep last detected state between frames
left_eye_flag = False
right_eye_flag = False
mouth_flag = False
all_landmarks = []

while True:
    success, image = video_cap.read()
    if not success:
        break

    image = cv2.resize(image, (800, 500))
    count += 1

    n = 5  # process every 5 frames
    if count % n == 0:
        left_eye_flag, right_eye_flag, mouth_flag, all_landmarks, left_ear, right_ear = process_image(image)
        # Debug print
        print(f"Left EAR: {left_ear:.3f} | Right EAR: {right_ear:.3f}")

        # Score update
        if left_eye_flag or right_eye_flag or mouth_flag:
            score += 1
        else:
            score = max(0, score - 1)

    # Draw landmarks based on last detection
    for landmarks in all_landmarks:
        left_eye = np.array(landmarks['left_eye'], np.int32).reshape((-1, 1, 2))
        right_eye = np.array(landmarks['right_eye'], np.int32).reshape((-1, 1, 2))
        draw_eyelid_outline(image, left_eye)
        draw_eyelid_outline(image, right_eye)

    # ==== Display Eye Status ====
    font = cv2.FONT_HERSHEY_SIMPLEX
    left_status = "Closed" if left_eye_flag else "Open"
    right_status = "Closed" if right_eye_flag else "Open"

    cv2.putText(image, f"Left Eye: {left_status}",
                (10, image.shape[0] - 50), font, 1,
                (0, 0, 255) if left_eye_flag else (0, 255, 0), 2)

    cv2.putText(image, f"Right Eye: {right_status}",
                (10, image.shape[0] - 10), font, 1,
                (0, 0, 255) if right_eye_flag else (0, 255, 0), 2)

    # ==== Display Score ====
    # cv2.putText(image, f"Score: {score}",
    #             (image.shape[1] - 200, image.shape[0] - 10), font, 1,
    #             (0, 0, 255), 2)

    # ==== Alert if Drowsy ====
    # cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
    # if score >= 5:
    #     cv2.putText(image, "Drowsy",
    #                 (image.shape[1] - 200, 50), font, 1,(0, 0, 255), 2)

    cv2.imshow('Drowsiness Detection', image)

    # Press any key to exit
    if cv2.waitKey(1) & 0xFF != 255:
        break

video_cap.release()
cv2.destroyAllWindows()
