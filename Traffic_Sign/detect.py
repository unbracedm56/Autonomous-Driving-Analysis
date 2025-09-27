import cv2
import numpy as np
import argparse
import os
import urllib.request

# --- Automatic Model File Downloader ---
def download_files():
    """Checks for model files and downloads them if they are missing."""
    files = {
        "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
        "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }

    for filename, url in files.items():
        if not os.path.exists(filename):
            print(f"[INFO] Downloading {filename}...")
            if filename == "yolov3.weights":
                print("[INFO] This is a large file (236 MB) and may take a few minutes...")
            
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"[INFO] {filename} downloaded successfully.")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                print("Please check your internet connection or download the file manually.")
                exit()
    print("[INFO] All model files are present.")

# --- Main Script Logic ---
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# Run the downloader first
download_files()

weights_path = "yolov3.weights"
config_path = "yolov3.cfg"
labels_path = "coco.names"

LABELS = open(labels_path).read().strip().split("\n")

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layerOutputs = net.forward(ln)

boxes = []
confidences = []
classIDs = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        if LABELS[classID] in ["traffic light", "stop sign"] and confidence > args["confidence"]:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

def get_traffic_light_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])
    
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    
    green_lower = np.array([40, 70, 70])
    green_upper = np.array([80, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.add(mask_red1, mask_red2)
    
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)

    if red_pixels > yellow_pixels and red_pixels > green_pixels:
        return "Red"
    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
        return "Yellow"
    elif green_pixels > red_pixels and green_pixels > yellow_pixels:
        return "Green"
    
    return "Unknown"

if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        
        color = (0, 255, 0)
        label = f"{LABELS[classIDs[i]]}: {confidences[i]:.2f}"
        
        if LABELS[classIDs[i]] == "traffic light":
            roi_y_start, roi_y_end = max(0, y), min(H, y + h)
            roi_x_start, roi_x_end = max(0, x), min(W, x + w)
            if roi_y_end > roi_y_start and roi_x_end > roi_x_start:
                roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                light_color = get_traffic_light_color(roi)
                label += f" ({light_color})"

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"[RESULT] {label}")

else:
    print("[INFO] No relevant objects (traffic lights or stop signs) detected.")

cv2.imwrite("output.jpg", image)
print("\n[INFO] Output image saved as output.jpg")

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

