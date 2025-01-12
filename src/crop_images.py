import os
import cv2
from tqdm import tqdm

def process_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

    for image_file in tqdm(image_files, desc="Processing images"):
        base_name = os.path.splitext(image_file)[0]
        label_id, file_id = base_name.split('_')[1:3]

        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image: {image_file}")
            continue

        original_output_path = os.path.join(output_dir, f"{base_name}_0.jpg")
        cv2.imwrite(original_output_path, image)

        annotation_file = os.path.join(input_dir, f"{base_name}.txt")
        if not os.path.exists(annotation_file):
            print(f"Annotation file not found for image: {image_file}")
            continue

        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines, start=1):
            parts = line.strip().split()
            class_id = parts[0]
            x_center, y_center, width, height = map(float, parts[1:])

            img_h, img_w = image.shape[:2]
            x_center, y_center, width, height = (
                x_center * img_w, y_center * img_h, width * img_w, height * img_h
            )
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            cropped_image = image[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
            crop_output_path = os.path.join(output_dir, f"image_{label_id}_{file_id}_{idx}.jpg")
            cv2.imwrite(crop_output_path, cropped_image)

    print(f"Processing complete. Images saved to {output_dir}.")


# Crop the train images
input_directory = "../datasets/train"
output_directory = "train"
process_images(input_directory, output_directory)

# Crop the dev images
input_directory = "../datasets/dev"
output_directory = "dev"
process_images(input_directory, output_directory)

# Crop the test images
# input_directory = "../datasets/test"
# output_directory = "test"
# process_images(input_directory, output_directory)
