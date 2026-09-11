"""
pipeline.py
Shared detection pipeline logic, extracted so both main.py (raw OpenCV window)
and app.py (Streamlit dashboard) can reuse the exact same processing code.
"""

import pickle
import os
import numpy as np
import cv2
import face_recognition
import mediapipe as mp

import sign_module
import expression_module

KNOWN_FACES_PATH = "registered_faces/known_faces.pkl"
FACE_MATCH_TOLERANCE = 0.5

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def load_known_faces():
    if not os.path.exists(KNOWN_FACES_PATH):
        return [], []
    with open(KNOWN_FACES_PATH, "rb") as f:
        known_faces = pickle.load(f)
    names = [entry["name"] for entry in known_faces]
    encodings = [entry["encoding"] for entry in known_faces]
    return names, encodings


def identify_face(encoding, known_names, known_encodings):
    if not known_encodings:
        return "Unknown"
    distances = face_recognition.face_distance(known_encodings, encoding)
    best_idx = int(np.argmin(distances))
    if distances[best_idx] <= FACE_MATCH_TOLERANCE:
        return known_names[best_idx]
    return "Unknown"


def process_frame(frame, frame_count, known_names, known_encodings, yolo_model,
                   state, face_every_n=5, object_every_n=3):
    """
    Runs all detection modules on one frame and draws results onto it.
    `state` is a dict carried across calls to cache face_boxes/object_boxes
    between the frame-skip intervals (mutated in place).

    Returns: (annotated_frame, results_dict)
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---------- FACE RECOGNITION ----------
    if frame_count % face_every_n == 0:
        face_boxes = []
        locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, locations)
        for (top, right, bottom, left), enc in zip(locations, encodings):
            label = identify_face(enc, known_names, known_encodings)
            face_boxes.append((top, right, bottom, left, label))
        state["face_boxes"] = face_boxes
    face_boxes = state.get("face_boxes", [])

    # ---------- OBJECT DETECTION ----------
    if frame_count % object_every_n == 0:
        object_boxes = []
        results = yolo_model(frame, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            if conf > 0.5:
                object_boxes.append((x1, y1, x2, y2, label, conf))
        state["object_boxes"] = object_boxes
    object_boxes = state.get("object_boxes", [])

    # ---------- EXPRESSION DETECTION ----------
    expression_text, _ = expression_module.process_frame(rgb_frame)

    # ---------- SIGN LANGUAGE RECOGNITION ----------
    gesture, hand_landmarks = sign_module.process_frame(rgb_frame)
    gesture_text = gesture if gesture else "No hand detected"
    if hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ---------- DRAW ----------
    for (top, right, bottom, left, label) in face_boxes:
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    for (x1, y1, x2, y2, label, conf) in object_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

    results = {
        "faces": [label for (_, _, _, _, label) in face_boxes],
        "objects": [label for (_, _, _, _, label, _) in object_boxes],
        "expression": expression_text,
        "gesture": gesture_text,
    }

    return frame, results
