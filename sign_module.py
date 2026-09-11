"""
sign_module.py
Hand landmark detection (MediaPipe) + rule-based classification of a small
predefined gesture set. This is the fastest path to a working "sign language"
demo without training a model. Extend GESTURES / classify_gesture() if you
have time to add more signs (e.g. train a small KNN on landmark vectors
instead of hardcoded rules).
"""

import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Landmark indices for fingertips and their corresponding PIP (middle) joints
TIP_IDS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP_IDS = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}


def _fingers_up(landmarks):
    """Return dict of finger_name -> bool (extended or not)."""
    fingers = {}
    # Thumb: compare x-coordinates (works for a roughly upright hand)
    fingers["thumb"] = landmarks[TIP_IDS["thumb"]].x < landmarks[PIP_IDS["thumb"]].x

    for name in ["index", "middle", "ring", "pinky"]:
        fingers[name] = landmarks[TIP_IDS[name]].y < landmarks[PIP_IDS[name]].y
    return fingers


def classify_gesture(landmarks):
    """
    Very simple rule-based classifier for a demo gesture set.
    Extend this dict-based logic for more signs if time allows.
    """
    f = _fingers_up(landmarks)
    up = [name for name, is_up in f.items() if is_up]

    if len(up) == 0:
        return "Stop / Fist"
    if len(up) == 5:
        return "Hello (Open Palm)"
    if up == ["thumb"]:
        return "Yes (Thumbs Up)"
    if set(up) == {"index", "middle"}:
        return "Peace"
    if up == ["index"]:
        return "One / Pointing"
    if set(up) == {"thumb", "pinky"}:
        return "Call Me"
    if set(up) == {"index", "pinky"}:
        return "I Love You"
    return "Unknown Gesture"


def process_frame(rgb_frame):
    """
    Takes an RGB frame, returns (gesture_text, hand_landmarks_for_drawing or None)
    """
    results = hands_detector.process(rgb_frame)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        gesture = classify_gesture(hand_landmarks.landmark)
        return gesture, hand_landmarks
    return None, None
