import streamlit as st

st.title("🚗 Vehicle Detection System")

st.subheader("Original Video")
uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file:
    st.video(uploaded_file)

st.subheader("Processed Output (YOLO Detection)")
st.video("output.mp4")  # your processed video
