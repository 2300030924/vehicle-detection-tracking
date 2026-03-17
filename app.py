import streamlit as st

st.title("🚗 Vehicle Detection System")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

if uploaded_file:
    st.video(uploaded_file)

st.write("⚠️ Detection demo version (UI working)")
