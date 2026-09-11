"""
expression_module.py
Facial expression detection using MediaPipe FaceMesh landmark geometry
instead of DeepFace. This avoids the TensorFlow/protobuf dependency
conflict with mediapipe entirely -- same package already used for hands,
no new install needed.

Heuristic-based (mouth shape, eyebrow position) rather than a trained
classifier. Good enough for a real-time demo; mention this design choice
if judges ask ("dependency-light, fully explainable, no GPU needed").
"""

import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

face_mesh_detector = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Key landmark indices
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
LEFT_EYEBROW = 105
LEFT_EYE_TOP = 159
FACE_LEFT = 234
FACE_RIGHT = 454
FACE_TOP = 10
FACE_BOTTOM = 152


def _dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def classify_expression(landmarks):
    lm = landmarks

    face_width = _dist(lm[FACE_LEFT], lm[FACE_RIGHT])
    face_height = _dist(lm[FACE_TOP], lm[FACE_BOTTOM])

    mouth_height = _dist(lm[UPPER_LIP], lm[LOWER_LIP])
    mouth_width = _dist(lm[LEFT_MOUTH_CORNER], lm[RIGHT_MOUTH_CORNER])

    mouth_center_y = (lm[UPPER_LIP].y + lm[LOWER_LIP].y) / 2
    corner_avg_y = (lm[LEFT_MOUTH_CORNER].y + lm[RIGHT_MOUTH_CORNER].y) / 2

    eyebrow_eye_gap = abs(lm[LEFT_EYE_TOP].y - lm[LEFT_EYEBROW].y) / face_height

    mouth_open_ratio = mouth_height / face_height
    smile_ratio = mouth_width / face_width
    corner_lift = (mouth_center_y - corner_avg_y) / face_height  # positive = corners raised

    # ---- classification rules (tune thresholds against your own face/lighting) ----
    if mouth_open_ratio > 0.06 and eyebrow_eye_gap > 0.05:
        return "Surprised"
    if corner_lift > 0.015 and smile_ratio > 0.42:
        return "Happy"
    if corner_lift < -0.01:
        return "Sad"
    if eyebrow_eye_gap < 0.028:
        return "Angry"
    return "Neutral"


def process_frame(rgb_frame):
    """
    Takes an RGB frame, returns (expression_text, face_landmarks_for_drawing or None)
    """
    results = face_mesh_detector.process(rgb_frame)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        expression = classify_expression(face_landmarks.landmark)
        return expression, face_landmarks
    return "No face detected", None
