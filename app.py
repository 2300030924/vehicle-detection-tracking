import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("🚗 Vehicle Detection System (YOLOv8)")

# Load YOLO model
model = YOLO("yolov8n.pt")  # small model (auto-downloads)

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

if uploaded_file:
    st.video(uploaded_file)

    st.write("⏳ Processing...")

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    results = model.predict(source=tfile.name, save=True)

    st.write("✅ Detection Completed")

    # Show output frames/images
    for r in results:
        if hasattr(r, "plot"):
            img = r.plot()
            st.image(img, channels="BGR")
