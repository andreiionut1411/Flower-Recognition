from ultralytics import YOLO


model = YOLO("yolov8m-oiv7.pt")

# Train the model
model.train(
    data="data.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    name="yolov8_flower_detector"
)