import os
import cv2
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_boxes(image_dir, output_dir, model):
	os.makedirs(output_dir, exist_ok=True)
	image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

	bounding_boxes_data = []

	for image_file in tqdm(image_files, desc="Processing images"):
		# We subtract 1 from the label to be 0-based as required by YOLO
		class_id = int(image_file.split("_")[1]) - 1
		image_path = os.path.join(image_dir, image_file)
		image = cv2.imread(image_path)
		img_height, img_width = image.shape[:2]

		results = model(image)

		# Prepare YOLO annotation file
		yolo_annotation_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}.txt")
		with open(yolo_annotation_path, "w") as yolo_file:
			for result in results:
				boxes = result.boxes.xyxy
				labels = result.boxes.cls
				scores = result.boxes.conf

				for i, box in enumerate(boxes):
					x1, y1, x2, y2 = box
					x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))

					x_center = ((x1 + x2) / 2) / img_width
					y_center = ((y1 + y2) / 2) / img_height
					width = (x2 - x1) / img_width
					height = (y2 - y1) / img_height
					yolo_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

	print(f"YOLO annotations saved in {output_dir}.")
	print(f"Images with bounding boxes saved in {output_dir}.")


def main():
	model = YOLO("yolov8m-oiv7.pt")

	# Generate bounding boxes for the test set
	image_dir = "test/"
	output_dir = "test_boxes/"
	generate_boxes(image_dir, output_dir, model)

	# Generate bounding boxes for the train set
	image_dir = "train/"
	output_dir = "train_boxes"
	generate_boxes(image_dir, output_dir, model)

	# Generate bounding boxes for the dev set
	image_dir = "dev"
	output_dir = "dev_boxes"
	generate_boxes(image_dir, output_dir, model)

if __name__ == '__main__':
    main()