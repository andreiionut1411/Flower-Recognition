import os
import sys
import cv2
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm


def compute_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1


def extract_label_from_filename(filename):
    try:
        label_id = int(filename.split('_')[1]) - 1
        return label_id
    except IndexError:
        return None


def process_images(test_directory, model):
    true_labels = []
    predicted_labels = []

    for filename in tqdm(os.listdir(test_directory)):
        if filename.endswith('.jpg'):
            image_path = os.path.join(test_directory, filename)
            true_label = extract_label_from_filename(filename)

            if not true_label:
                print(f"Warning: Filename {filename} is incorrectly formatted.")
                continue

            image = cv2.imread(image_path)
            results = model(image)

            for result in results:
                boxes = result.boxes.xyxy
                labels = result.boxes.cls
                names = result.names

                for i, box in enumerate(boxes):
                    label = int(labels[i])
                    predicted_labels.append(label)
                    true_labels.append(true_label)

    # Calculate overall metrics
    accuracy, precision, recall, f1 = compute_metrics(true_labels, predicted_labels)

    return accuracy, precision, recall, f1


model = YOLO("detect/yolov8_flower_detector/weights/best.pt")
use_original_metrics = False

if use_original_metrics:
    metrics = model.val(data="data.yaml", split="test")
    print(metrics)
else:
    test_directory = '../datasets/test'
    model = YOLO("detect/yolov8_flower_detector/weights/best.pt")
    accuracy, precision, recall, f1 = process_images(test_directory, model)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")