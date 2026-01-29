from ultralytics import YOLO
import cv2
import os
from tracker import CentroidTracker

# Load trained YOLO model
model = YOLO("runs/detect/vehicle_detection/weights/best.pt")

# Initialize tracker
tracker = CentroidTracker(max_disappeared=40)

# Counting line position (Y-axis)
LINE_Y = 300

# Vehicle count
total_count = 0
counted_ids = set()

# Use validation images as frames (acts like video)
image_folder = "dataset/val/images"
image_files = sorted(os.listdir(image_folder))

for image_name in image_files:
    image_path = os.path.join(image_folder, image_name)
    frame = cv2.imread(image_path)

    results = model(frame, conf=0.4)

    rects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            rects.append((x1, y1, x2, y2))

    objects = tracker.update(rects)

    # Draw counting line
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (255, 0, 0), 2)

    # Draw bounding boxes
    for (x1, y1, x2, y2) in rects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw IDs and count vehicles
    for objectID, centroid in objects.items():
        cX, cY = centroid

        # Count vehicle if it crosses the line
        if objectID not in counted_ids and cY > LINE_Y:
            counted_ids.add(objectID)
            total_count += 1

        cv2.putText(
            frame,
            f"ID {objectID}",
            (cX - 10, cY - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
        cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

    # Display count
    cv2.putText(
        frame,
        f"Vehicle Count: {total_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    cv2.imshow("Vehicle Detection, Tracking & Counting", frame)
    key = cv2.waitKey(300)
    if key == 27:
        break

cv2.destroyAllWindows()
