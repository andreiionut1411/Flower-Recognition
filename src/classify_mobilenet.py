import sys
import torch
import torch.nn as nn
import cv2
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image
import numpy as np
import pandas as pd


def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <cats/dogs> <image_path> <output_path>")
        sys.exit(1)

    toxicity_type = sys.argv[1]
    image_path = sys.argv[2]
    output_path = sys.argv[3]

    if toxicity_type not in ['cats', 'dogs']:
        print("Error: Argument must be 'cats' or 'dogs'")
        sys.exit(1)

    toxicity_type = "toxic_to_" + toxicity_type
    yolo_model = YOLO("yolov8m-oiv7.pt")

    num_classes = 102
    classifier_model = mobilenet_v3_large(pretrained=False)
    classifier_model.classifier[3] = nn.Linear(classifier_model.classifier[3].in_features, num_classes)
    classifier_model.load_state_dict(torch.load("mobilenetv3_flower_classifier.pth"))
    classifier_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_model.to(device)

    flower_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = cv2.imread(image_path)
    original_image = image.copy()
    image_height, image_width = image.shape[:2]

    toxic_data = pd.read_csv("toxic_plants.csv")
    plant_names = toxic_data["plant_name"].tolist()

    # Perform detection with YOLO
    results = yolo_model(image)
    detections = results[0]
    names = results[0].names

    for detection in detections.boxes:
        cls_id = int(detection.cls.cpu().numpy())
        label = names[cls_id]
        if label not in ["Common sunflower", "Flower", "Plant", "Rose"]:
            continue

        x1, y1, x2, y2 = map(int, detection.xyxy.cpu().numpy().flatten())
        cropped_image = image[y1:y2, x1:x2]
        cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        input_tensor = flower_transforms(cropped_image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = classifier_model(input_tensor)
            _, predicted_class = torch.max(outputs, 1)
            predicted_index = predicted_class.item()
            class_name = plant_names[predicted_index]

        toxicity_info = toxic_data[toxic_data['plant_name'] == class_name]

        if int(toxicity_info[toxicity_type].values[0]) == 0:
            toxicity_string = "Non-Toxic"
        else:
            toxicity_string = "Toxic"

        color = (0, 255, 0) if int(toxicity_info[toxicity_type].values[0]) == 0 else (0, 0, 255)
        cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 4)
        (label_width, label_height), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)

        if y1 - label_height - 10 < 0:
            label_y = y1 + label_height + 10
        else:
            label_y = y1 - 10

        if x1 + label_width + 10 > original_image.shape[1]:
            label_x = x1 - label_width - 10
        else:
            label_x = x1

        cv2.putText(original_image, f"{class_name}: {toxicity_string}", (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    cv2.imwrite(output_path, original_image)
    print(f"Processed image saved to {output_path}")


if __name__ == "__main__":
    main()
