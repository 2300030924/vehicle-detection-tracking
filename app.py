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
