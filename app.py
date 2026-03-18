<<<<<<< HEAD
import streamlit as st
import tempfile
import time
from test import process_video

# Page config
st.set_page_config(
    page_title="Vehicle Detection Dashboard",
    page_icon="🚗",
    layout="wide"
)

# ---------- HEADER ----------
st.title("🚗 Vehicle Detection, Tracking & Counting")
st.markdown("""
### AI-powered traffic analysis system  
Upload a video to detect, track, and count vehicles in real time.

⚡ *Tip: Use short videos (5–10 seconds) for faster results*
""")

st.warning("💡 You can upload your own video OR try the demo video below")

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
iou = st.sidebar.slider("IOU Threshold", 0.1, 1.0, 0.7)

st.sidebar.markdown("---")
st.sidebar.info("This app uses YOLOv8 + Centroid Tracking")

# ---------- DEMO VIDEO SECTION ----------
st.markdown("### 🎬 Try Demo Video")

if st.button("▶️ Run Demo Video"):
    sample_path = "sample.mp4"
    output_path = "output.mp4"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📥 Demo Input")
        st.video(sample_path)

    # Progress
    progress = st.progress(0)
    status = st.empty()

    status.text("🔍 Initializing model...")
    time.sleep(1)
    progress.progress(20)

    status.text("🚗 Detecting vehicles...")
    time.sleep(1)
    progress.progress(50)

    process_video(sample_path, output_path)

    progress.progress(100)
    status.text("✅ Processing Complete!")

    st.success("🎉 Demo Completed!")

    with col2:
        st.subheader("📤 Output Video")
        st.video(output_path)

# ---------- FILE UPLOAD ----------
uploaded_video = st.file_uploader("📤 Upload Video", type=["mp4"])

if uploaded_video:

    col1, col2 = st.columns(2)

    # ---------- INPUT VIDEO ----------
    with col1:
        st.subheader("📥 Input Video")
        st.video(uploaded_video)

    # Save video temporarily
    temp_input = tempfile.NamedTemporaryFile(delete=False)
    temp_input.write(uploaded_video.read())

    # ---------- PROGRESS BAR ----------
    progress = st.progress(0)
    status = st.empty()

    status.text("🔍 Initializing model...")
    time.sleep(1)
    progress.progress(20)

    status.text("🚗 Detecting vehicles...")
    time.sleep(1)
    progress.progress(50)

    output_path = "output.mp4"

    # ---------- RUN MODEL ----------
    process_video(temp_input.name, output_path)

    progress.progress(100)
    status.text("✅ Processing Complete!")

    st.success("🎉 Detection Finished Successfully!")

    # ---------- OUTPUT VIDEO ----------
    with col2:
        st.subheader("📤 Output Video")
        st.video(output_path)

    # ---------- METRICS ----------
    st.markdown("### 📊 Summary")

    col3, col4, col5 = st.columns(3)

    col3.metric("Model", "YOLOv8")
    col4.metric("Tracking", "Centroid")
    col5.metric("Status", "Completed ✅")

    # ---------- DOWNLOAD ----------
    with open(output_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Processed Video",
            data=f,
            file_name="vehicle_detection_output.mp4",
            mime="video/mp4"
        )

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("👩‍💻 Developed by Bhavya Sri | 🚀 AI & Computer Vision Project")
=======
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import tempfile

app = Flask(__name__)

model = YOLO("yolov8n.pt")

@app.route('/')
def home():
    return "🚗 Vehicle Detection API Running!"

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    file.save(temp_file.name)

    cap = cv2.VideoCapture(temp_file.name)

    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        count += len(results[0].boxes)

    cap.release()

    return jsonify({"vehicle_count": count})

if __name__ == "__main__":
    app.run()
>>>>>>> de04d38e7243f89bc8c3996497b13d0a46597354
