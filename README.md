# 🚗 Vehicle Detection, Tracking & Counting using YOLOv8

This project implements an end-to-end vehicle detection, tracking, and counting system using YOLOv8 and OpenCV. It assigns unique IDs to vehicles and counts them when they cross a virtual counting line, ensuring accurate and non-duplicate counting.

The system simulates real-world traffic monitoring applications such as traffic flow analysis and intelligent transportation systems.

---

## 🚗 Demo

![Vehicle Detection Output](https://github.com/user-attachments/assets/63c78fb8-0ec5-4ad6-b3c7-d7f87cf6a0e9)

---

## ✨ Features

- Real-time vehicle detection using YOLOv8
- Multi-object tracking with unique ID assignment
- Line-crossing logic for accurate vehicle counting
- Annotated output with bounding boxes, IDs, and total count
- End-to-end deployable computer vision pipeline

---

## 🛠 Tech Stack

- Python  
- YOLOv8 (Ultralytics)  
- OpenCV  
- NumPy  
- Streamlit (for deployment)

---

## ⚙️ How It Works

1. Vehicles are detected in each frame using YOLOv8  
2. Detected objects are tracked using centroid-based tracking  
3. Each vehicle is assigned a unique ID across frames  
4. Vehicles are counted when they cross a predefined virtual line  
5. The output video displays bounding boxes, IDs, and total count  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
