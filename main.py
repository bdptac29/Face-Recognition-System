"""
main.py (v3 - multi-person face recognition)
AI-Powered Real-Time Recognition & Accessibility System
Combines: Face Recognition (multi-person, named), Object Detection (YOLOv8),
Expression Detection (MediaPipe FaceMesh heuristics), and Sign Language
Recognition (MediaPipe Hands).

Run register_face.py once PER PERSON before this.
"""

import cv2
import pickle
import os
import numpy as np
import face_recognition
from ultralytics import YOLO
import mediapipe as mp

import sign_module
import expression_module

# ---------- CONFIG ----------
KNOWN_FACES_PATH = "registered_faces/known_faces.pkl"
FACE_MATCH_TOLERANCE = 0.5

FACE_EVERY_N = 5
OBJECT_EVERY_N = 3

YOLO_MODEL = "yolov8n.pt"
# ----------------------------

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


def load_known_faces():
    if not os.path.exists(KNOWN_FACES_PATH):
        print("WARNING: No registered faces found. Run register_face.py first.")
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


def main():
    known_names, known_encodings = load_known_faces()
    print(f"Loaded {len(known_names)} registered face(s): {known_names}")

    print("Loading YOLOv8 model...")
    yolo_model = YOLO(YOLO_MODEL)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    frame_count = 0
    face_boxes = []
    object_boxes = []

    print("Starting main loop. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---------- FACE RECOGNITION (multi-person) ----------
        if frame_count % FACE_EVERY_N == 0:
            face_boxes = []
            locations = face_recognition.face_locations(rgb_frame)
            encodings = face_recognition.face_encodings(rgb_frame, locations)

            for (top, right, bottom, left), enc in zip(locations, encodings):
                label = identify_face(enc, known_names, known_encodings)
                face_boxes.append((top, right, bottom, left, label))

        # ---------- OBJECT DETECTION ----------
        if frame_count % OBJECT_EVERY_N == 0:
            object_boxes = []
            results = yolo_model(frame, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id]
                if conf > 0.5:
                    object_boxes.append((x1, y1, x2, y2, label, conf))

        # ---------- EXPRESSION DETECTION ----------
        expression_text, face_landmarks = expression_module.process_frame(rgb_frame)

        # ---------- SIGN LANGUAGE RECOGNITION ----------
        gesture, hand_landmarks = sign_module.process_frame(rgb_frame)
        gesture_text = gesture if gesture else "No hand detected"
        if hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # ---------- DRAW EVERYTHING ----------
        for (top, right, bottom, left, label) in face_boxes:
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for (x1, y1, x2, y2, label, conf) in object_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

        cv2.putText(frame, f"Emotion: {expression_text}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Sign: {gesture_text}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        cv2.imshow("AI-Powered Real-Time Recognition & Accessibility System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
