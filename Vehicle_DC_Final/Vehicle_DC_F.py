import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from ultralytics import YOLO
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download

IMAGE_SIZE = (224, 224)
REPO_ID = "Coder-M/Vehicle"
vehicle_ids = [2, 3, 5, 7]

model_path = snapshot_download(repo_id="Coder-M/Vehicle")
model = tf.keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")

yolo_model = YOLO("yolov8n.pt")

class_names = [
    'Auto',
    'Bus',
    'Empty road',
    'Motorcycles',
    'Tempo Traveller',
    'Tractor',
    'Truck',
    'cars'
]

def classify_from_input():
    img_path = input("Enter the path of the image: ").strip()
    output_path = classify_vehicles(img_path, save_path="classified_output.jpg")
    print(f"Classified image saved to: {output_path}")

def classify_vehicles(image_path, save_path="output.jpg"):
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Could not load image: {image_path}")

    results = yolo_model(image_path)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    for box, cls_id in zip(boxes, classes):
        if cls_id not in vehicle_ids:
            continue

        x1, y1, x2, y2 = map(int, box[:4])
        crop = orig_img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, IMAGE_SIZE)
        crop_input = preprocess_input(np.expand_dims(crop_resized.astype("float32"), axis=0))

        preds = model(crop_input, training=False)
        preds = preds["output_0"].numpy()

        pred_label = class_names[np.argmax(preds)]

        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig_img, pred_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imwrite(save_path, orig_img)
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

    return save_path

classify_from_input()