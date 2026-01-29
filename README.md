# AI-Based Vehicle Detection, Tracking and Counting System

This project implements a computer vision-based system to detect, track, label, and count vehicles using YOLO and tracking algorithms.
“I used a pretrained YOLOv8n model provided by Ultralytics and fine-tuned it on a custom car-only dataset using bounding box annotations. The training process adapts the model specifically for vehicle detection in traffic scenes. After training completes, a fine-tuned model (best.pt) is generated, which I use for detection, tracking, and vehicle counting in my project.”

Like yolo has more than 70 casses , now iam finetuning it only for one class:car

You can remember it like this:

1️⃣ Took a pretrained YOLOv8 model from Ultralytics
2️⃣ Used transfer learning
3️⃣ Provided a custom car-only dataset
4️⃣ Fine-tuned the model using bounding box annotations
5️⃣ Produced a custom trained model (best.pt) for car detection


This project implements an AI-based vehicle detection, tracking, and counting system using YOLOv8.
The system detects cars from traffic images/video, assigns unique IDs to each vehicle, and accurately counts vehicles as they cross a virtual counting line.

The project demonstrates a complete computer vision pipeline, starting from dataset preparation and model training to real-time vehicle counting.

🎯 Objectives of the Project

Detect only cars in traffic scenes

Track each car across multiple frames using a unique ID

Count each vehicle only once

Avoid double counting

Build a real-world traffic monitoring solution

🧠 Technologies & Tools Used

Python

YOLOv8 (Ultralytics)

OpenCV

NumPy

SciPy

VS Code