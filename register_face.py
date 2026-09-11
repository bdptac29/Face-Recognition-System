"""
register_face.py (multi-person)
Run this once PER PERSON you want recognized. Enter their name, then press
'c' to capture and save their face. Encodings are appended to a shared file
so you can register as many people as you like without overwriting anyone.
"""

import cv2
import face_recognition
import pickle
import os

SAVE_PATH = "registered_faces/known_faces.pkl"


def load_known_faces():
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "rb") as f:
            return pickle.load(f)
    return []  # list of {"name": str, "encoding": ndarray}


def save_known_faces(known_faces):
    os.makedirs("registered_faces", exist_ok=True)
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(known_faces, f)


def main():
    name = input("Enter this person's name: ").strip()
    if not name:
        print("Name cannot be empty. Aborting.")
        return

    known_faces = load_known_faces()
    cap = cv2.VideoCapture(0)

    print(f"Registering '{name}'. Press 'c' to capture, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read from webcam.")
            break

        display = frame.copy()
        cv2.putText(display, f"Registering: {name} | 'c' capture, 'q' quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Register Face", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if len(face_locations) == 0:
                print("No face detected. Try again with better lighting/positioning.")
                continue

            encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            encoding = encodings[0]

            known_faces.append({"name": name, "encoding": encoding})
            save_known_faces(known_faces)

            print(f"'{name}' registered. Total people registered: {len(known_faces)}")
            break

        elif key == ord('q'):
            print("Cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
