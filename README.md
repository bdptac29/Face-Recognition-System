# AI-Powered Real-Time Recognition & Accessibility System

## Setup

```bash
pip install -r requirements.txt
```

Note: `face_recognition` depends on `dlib`. If it fails to install:
- **Windows**: install via `conda install -c conda-forge dlib` or grab a prebuilt wheel for your Python version.
- **Mac/Linux**: `pip install cmake` first, then retry `pip install dlib`.
- **If you're truly stuck on time**: swap face recognition to MediaPipe Face Detection + a simple face-crop histogram comparison, or use OpenCV's built-in `cv2.face.LBPHFaceRecognizer`. Ask if you need this fallback written.

The first run of `main.py` will auto-download `yolov8n.pt` (~6MB) and the DeepFace emotion model weights — do this ONCE early, before your demo, since it needs internet.

## Run order

1. `python register_face.py` — press `c` to capture and save your face. Do this once.
2. `python main.py` — starts the full pipeline. Press `q` to quit.

## What's implemented

- **Face Recognition**: `face_recognition` library, compares live face encodings against your registered encoding.
- **Object Detection**: YOLOv8n (pretrained on COCO, 80 classes), runs every 3rd frame for speed.
- **Expression Detection**: DeepFace emotion model, runs every 5th frame.
- **Sign Language Recognition**: MediaPipe Hands landmarks + rule-based classifier for a demo gesture set (Hello, Yes, Peace, Stop, One, Call Me, I Love You). Runs every frame since gestures are the interactive part.

## Team split (2 people, few hours)

- **Person A**: Test/tune face recognition + expression detection. Try different `FACE_MATCH_TOLERANCE` values, handle multiple faces, improve the "Unknown" vs registered-user display.
- **Person B**: Test/tune object detection (maybe filter to relevant classes only — e.g. only show "person, bottle, chair, phone" instead of all 80) + expand the sign gesture set in `sign_module.py`.
- **Merge**: run `main.py` together, fix FPS issues by adjusting the `_EVERY_N` constants, polish the on-screen UI.

## If you have extra time

- Replace rule-based sign classification with a trained classifier: record ~30 samples per gesture (landmark coordinates from `sign_module.py`), train a quick `sklearn` KNN or small MLP — looks more "ML" to judges than hardcoded rules.
- Add a simple Tkinter/Streamlit overlay instead of raw OpenCV window for a nicer demo UI.
- Add text-to-speech (`pyttsx3`) so recognized signs are spoken aloud — strong accessibility narrative for judges.
- Log recognized emotions/objects/signs to a sidebar or file as a "session summary."

## Performance tips

- If FPS is low: increase the `_EVERY_N` frame-skip constants, or downscale the frame (`cv2.resize`) before running YOLO/DeepFace, then scale bounding boxes back up.
- GPU: ultralytics and DeepFace(TensorFlow) will use CUDA automatically if available — just make sure `torch` and `tensorflow` are GPU-enabled builds if you want the speed boost.
