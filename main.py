from ultralytics import YOLO

def train_model():
    # Load a pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # Train the model on your dataset
    model.train(
        data="dataset/data.yaml",
        epochs=20,
        imgsz=640,
        batch=8,
        name="vehicle_detection"
    )

if __name__ == "__main__":
    train_model()
