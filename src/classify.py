import sys
import cv2
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <cats/dogs> <image_path>")
        sys.exit(1)

    toxicity_type = sys.argv[1]
    image_path = sys.argv[2]

    model = YOLO("detect/yolov8_flower_detector/weights/best.pt")
    toxic_data = pd.read_csv("toxic_plants.csv")

    if toxicity_type not in ['cats', 'dogs']:
        print("Error: Argument must be 'cats' or 'dogs'")
        sys.exit(1)

    toxicity_type = "toxic_to_" + toxicity_type
    image = cv2.imread(image_path)
    results = model(image)

    for result in results:
        boxes = result.boxes.xyxy
        labels = result.boxes.cls
        names = result.names

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = names[int(labels[i])]
            toxicity_info = toxic_data[toxic_data['plant_name'] == label]

            if toxicity_info.empty:
                print(f"Warning: Plant name '{label}' not found in toxicity data.")
                continue

            toxic_to = toxicity_info[toxicity_type].values[0]
            color = (0, 255, 0) if toxic_to == 0 else (0, 0, 255)

            if int(toxicity_info[toxicity_type].values[0]) == 0:
                toxicity_string = "Non-Toxic"
            else:
                toxicity_string = "Toxic"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
            (label_width, label_height), baseline = cv2.getTextSize(f"{label}: {toxicity_string}",
                                                         cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)

            # Check if the label goes off the top of the image
            if y1 - label_height - 10 < 0:
                label_y = y1 + label_height + 10
            else:
                label_y = y1 - 10

            # Check if the label goes off the left of the image
            if x1 + label_width + 10 > image.shape[1]:
                label_x = x1 - label_width - 10
            else:
                label_x = x1


            cv2.putText(image, f"{label}: {toxicity_string}",
                        (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    output_image_path = "output_image_with_boxes.jpg"  # Path where you want to save the image
    cv2.imwrite(output_image_path, image)
    print(f"Image saved to {output_image_path}")

if __name__ == "__main__":
    main()
