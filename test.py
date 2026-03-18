from ultralytics import YOLO
import cv2
from tracker import CentroidTracker

def process_video(input_path, output_path):
    model = YOLO("yolov8n.pt")   # lightweight model
    vehicle_classes = [1, 2, 3, 5, 7]

    tracker = CentroidTracker(max_disappeared=10)

    LINE_Y = 100   # 🔥 TOP POSITION (fixed here only once)
    total_count = 0
    counted_ids = set()

    cap = cv2.VideoCapture(input_path)

    width = 640
    height = 480

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height), True)  # lower FPS

    frame_count = 0  # 🔥 for skipping frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 🔥 SPEED BOOST: process only every 5th frame
        if frame_count % 5 != 0:
            continue

        # 🔥 resize for faster processing
        frame = cv2.resize(frame, (640, 480))

        results = model(frame, conf=0.5, iou=0.7)

        rects = []

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])

                if cls in vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    area = (x2 - x1) * (y2 - y1)

                    if area > 3000:
                        rects.append((x1, y1, x2, y2))

                        class_name = model.names[cls]
                        cv2.putText(frame, class_name, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 2)

        objects = tracker.update(rects)

        # 🔥 draw line (thicker for visibility)
        cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (255, 0, 0), 4)

        # draw bounding boxes
        for (x1, y1, x2, y2) in rects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 🔥 tracking + counting (robust)
        for objectID, centroid in objects.items():
            cX, cY = centroid

            # ✅ buffer zone to avoid missed counts
            if objectID not in counted_ids and (LINE_Y - 20) < cY < (LINE_Y + 20):
                counted_ids.add(objectID)
                total_count += 1

            cv2.putText(frame, f"ID {objectID}", (cX - 10, cY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

        # display total count
        cv2.putText(frame, f"Vehicle Count: {total_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        out.write(frame)

    cap.release()
    out.release()