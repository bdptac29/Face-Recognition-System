"""
app.py
Streamlit dashboard frontend for the AI-Powered Real-Time Recognition &
Accessibility System. Run with: streamlit run app.py
"""

import streamlit as st
import cv2
from ultralytics import YOLO

import pipeline

st.set_page_config(
    page_title="AI Recognition & Accessibility System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM STYLING ----------
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .block-container {
        padding-top: 2rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #fafafa;
        margin-bottom: 0.1rem;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #9aa0a6;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .status-live {
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        border: 1px solid #2ecc71;
    }
    .status-offline {
        background-color: rgba(148, 163, 184, 0.15);
        color: #94a3b8;
        border: 1px solid #94a3b8;
    }
    .metric-card {
        background-color: #1a1d24;
        border: 1px solid #2a2e37;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9aa0a6;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #fafafa;
    }
    .chip {
        display: inline-block;
        background-color: #262b36;
        color: #e2e8f0;
        border: 1px solid #384152;
        border-radius: 16px;
        padding: 3px 12px;
        margin: 3px 4px 3px 0;
        font-size: 0.82rem;
    }
    .chip-active {
        background-color: rgba(99, 179, 237, 0.15);
        border-color: #63b3ed;
        color: #63b3ed;
    }
    .user-chip {
        display: inline-block;
        background-color: rgba(99, 179, 237, 0.1);
        border: 1px solid #2a2e37;
        border-radius: 8px;
        padding: 6px 12px;
        margin-bottom: 6px;
        font-size: 0.85rem;
        color: #e2e8f0;
        width: 100%;
    }
    section[data-testid="stSidebar"] {
        background-color: #12151c;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_yolo_model():
    return YOLO("yolov8n.pt")


@st.cache_resource
def get_known_faces():
    return pipeline.load_known_faces()


yolo_model = get_yolo_model()
known_names, known_encodings = get_known_faces()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 👤 Registered Users")
    if known_names:
        for name in known_names:
            st.markdown(f"<div class='user-chip'>🟢 {name}</div>", unsafe_allow_html=True)
    else:
        st.warning("No faces registered. Run register_face.py first.")

    st.divider()
    st.markdown("### ⚙️ Controls")
    run = st.checkbox("Start Camera", value=False, key="Start Camera")

    st.divider()
    st.markdown("### 🧠 Tech Stack")
    st.caption("Face Recognition · YOLOv8 · MediaPipe FaceMesh · MediaPipe Hands")

# ---------- HEADER ----------
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.markdown('<div class="hero-title">AI-Powered Real-Time Recognition & Accessibility System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Face Recognition · Object Detection · Expression Detection · Sign Language Recognition</div>', unsafe_allow_html=True)
with header_col2:
    if run:
        st.markdown('<div class="status-badge status-live">● LIVE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-offline">● OFFLINE</div>', unsafe_allow_html=True)

# ---------- MAIN LAYOUT ----------
video_col, stats_col = st.columns([2.2, 1])

with video_col:
    video_placeholder = st.empty()
    if not run:
        video_placeholder.info("Enable 'Start Camera' in the sidebar to begin the live feed.")

with stats_col:
    st.markdown("#### Live Recognition Panel")
    face_metric = st.empty()
    expression_metric = st.empty()
    gesture_metric = st.empty()
    st.markdown("#### Detected Objects")
    objects_area = st.empty()


def render_metric_card(placeholder, label, value):
    placeholder.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_objects(placeholder, objects):
    if not objects:
        placeholder.markdown("<span class='chip'>None detected</span>", unsafe_allow_html=True)
        return
    chips = "".join(f"<span class='chip chip-active'>{obj}</span>" for obj in sorted(set(objects)))
    placeholder.markdown(chips, unsafe_allow_html=True)


# Render initial empty state
render_metric_card(face_metric, "Recognized Face(s)", "—")
render_metric_card(expression_metric, "Expression", "—")
render_metric_card(gesture_metric, "Sign Detected", "—")
render_objects(objects_area, [])

# ---------- CAMERA LOOP ----------
if run:
    cap = cv2.VideoCapture(0)
    frame_count = 0
    state = {}

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Could not read from webcam.")
            break

        frame_count += 1
        annotated_frame, results = pipeline.process_frame(
            frame, frame_count, known_names, known_encodings, yolo_model, state
        )

        video_placeholder.image(
            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )

        faces_text = ", ".join(results["faces"]) if results["faces"] else "None detected"
        render_metric_card(face_metric, "Recognized Face(s)", faces_text)
        render_metric_card(expression_metric, "Expression", results["expression"])
        render_metric_card(gesture_metric, "Sign Detected", results["gesture"])
        render_objects(objects_area, results["objects"])

        run = st.session_state.get("Start Camera", run)

    cap.release()
