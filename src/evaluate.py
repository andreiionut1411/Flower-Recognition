from ultralytics import YOLO

model = YOLO("detect/yolov8_flower_detector/weights/best.pt")
metrics = model.val(data="data.yaml", split="test")

print(metrics)
