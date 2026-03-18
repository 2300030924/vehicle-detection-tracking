# 🚗Vehicle Detection, Tracking & Counting using YOLOv8

This project implements a real-time vehicle detection, tracking, and counting
system using YOLOv8 and OpenCV. It assigns unique IDs to vehicles and counts them
when they cross a virtual counting line, avoiding double counting.

The system simulates real-world traffic monitoring use cases such as traffic
flow analysis and intelligent transportation systems.

## Demo

![Output Screenshot](<img width="775" height="799" alt="Screenshot 2026-03-18 095549" src="https://github.com/user-attachments/assets/63c78fb8-0ec5-4ad6-b3c7-d7f87cf6a0e9" />
)
## Features
- YOLOv8-based real-time vehicle detection
- Unique ID assignment for multi-object tracking
- Virtual counting line logic to avoid duplicate counts
- Annotated output video with bounding boxes, IDs, and count
- End-to-end computer vision pipeline

## Tech Stack
- Python
- YOLOv8
- OpenCV
- NumPy

## How it Works
1. Vehicles are detected frame-by-frame using a YOLOv8 model
2. Each detected vehicle is assigned a unique tracking ID
3. Vehicles are counted when they cross a predefined virtual line
4. An annotated output video is generated with the live count



## How to Run
```bash
pip install -r requirements.txt
python test.py
