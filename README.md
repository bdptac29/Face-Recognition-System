# AI-Powered Real-Time Recognition & Accessibility System

An AI-powered computer vision system that uses a webcam to understand and interpret a person's face, gestures, and surrounding environment in real time. It combines face recognition, object detection, facial expression analysis, and sign language recognition into a single live pipeline — built as a demonstration of how computer vision can improve accessibility and communication for people with hearing or speech difficulties.

## Features

- **Face Recognition** — Identifies registered users by name in real time using face embeddings.
- **Object Recognition** — Detects and labels objects in the surrounding environment using YOLOv8.
- **Expression Detection** — Classifies basic facial expressions (Happy, Sad, Angry, Surprised, Neutral) using MediaPipe FaceMesh landmark geometry.
- **Sign Language Recognition** — Recognizes a predefined set of hand gestures (Hello, Yes, Peace, Stop, One, Call Me, I Love You) and converts them into on-screen text using MediaPipe Hands.
- **Multi-person support** — Register as many users as needed, each identified individually.
- **Two interfaces** — A lightweight OpenCV window for quick testing, and a polished Streamlit dashboard for demos.

## Tech Stack

Python · OpenCV · MediaPipe · YOLOv8 (Ultralytics) · face_recognition · Streamlit

## Project Structure
├── main.py # Raw OpenCV window pipeline
├── app.py # Streamlit dashboard frontend
├── pipeline.py # Shared detection logic used by both interfaces
├── register_face.py # Registers a new person's face (run once per person)
├── list_faces.py # Lists all currently registered faces
├── sign_module.py # Hand landmark detection + gesture classification
├── expression_module.py # Facial landmark-based expression classification
├── requirements.txt
└── registered_faces/ # Stores registered face encodings (auto-created)


## Setup

```bash
pip install -r requirements.txt
pip install streamlit
```

> **Note:** `face_recognition` depends on `dlib`, which can fail to build on Windows without a C++ compiler. If installation fails, use `pip install dlib-bin` followed by `pip install face_recognition --no-deps`.

## Usage

### 1. Register faces
Run once per person you want the system to recognize:
```bash
python register_face.py
```
Enter their name, then press `c` to capture their face (`q` to cancel).

### 2. Check registered faces (optional)
```bash
python list_faces.py
```

### 3. Run the system

**Option A — OpenCV window:**
```bash
python main.py
```
Press `q` to quit.

**Option B — Streamlit dashboard (recommended for demos):**
```bash
streamlit run app.py
```
Opens a browser dashboard at `localhost:8501`. Check "Start Camera" in the sidebar to begin.

## How It Works

Each frame from the webcam is processed through four independent modules:

1. **Face recognition** compares detected faces against registered encodings and labels them by name (or "Unknown").
2. **Object detection** runs a pretrained YOLOv8 model to draw bounding boxes around recognized objects.
3. **Expression detection** measures mouth shape and eyebrow position from facial landmarks to classify the dominant expression.
4. **Sign recognition** tracks hand landmarks and classifies finger positions against a small predefined gesture set.

To maintain real-time performance, face recognition and object detection run on a frame-skip interval rather than every single frame, while expression and gesture detection (both lightweight MediaPipe operations) run every frame for responsiveness.

## Applications

- Accessibility and assistive technology
- Smart classrooms
- Human-computer interaction
- Communication tools for sign-language users
- Smart cameras and security systems

## Future Improvements

- Replace rule-based sign classification with a trained classifier (KNN/MLP on landmark vectors) for a larger gesture vocabulary
- Add text-to-speech output for recognized signs
- Session logging of detected emotions/objects/signs over time
- Support for full ASL alphabet recognition
