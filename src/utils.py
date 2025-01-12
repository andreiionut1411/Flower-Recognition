from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load pretrained YOLOv8 model (for object detection)
model = YOLO("yolov8m-oiv7.pt")  # 'yolov8n.pt' is a smaller model, use 'yolov8.pt' for higher accuracy

# Read input image
image_path = "my_img/WhatsApp Image 2025-01-09 at 22.23.31.jpeg"
image = cv2.imread(image_path)

# Perform inference on the image
results = model(image)

# Access the first result in the list and visualize results (bounding boxes and labels on the image)
results[0].show()  # Show the image with bounding boxes and labels

# Optionally, you can save the image with bounding boxes
output_image = results[0].plot()  # Get the image with bounding boxes
cv2.imwrite("output_image_with_boxes.jpg", output_image)

# To show the image inline (useful in Jupyter notebooks)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
