from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8m-oiv7.pt")

# Train the model
model.train(
    data="data.yaml",  # Path to the data.yaml file
    epochs=10,         # Number of training epochs
    imgsz=640,         # Image size for training
    batch=8,          # Batch size
    name="yolov8_flower_detector"  # Save the model under this name
)