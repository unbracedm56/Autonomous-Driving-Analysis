# Fix for macOS threading issues - MUST BE FIRST
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import streamlit as st
import cv2
cv2.setNumThreads(0)
import numpy as np
import pandas as pd
import tempfile
import sys

# Add paths for custom modules
sys.path.append('Lane_detection')
sys.path.append('Traffic_Sign')
sys.path.append('Vehicle_DC_Final')
sys.path.append('Pedestrian_detection')

from tensorflow.keras.models import load_model
from Lane_detection.custom_layers import spatial_attention, weighted_bce
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(page_title="Autonomous Driving Analysis System", page_icon="🚗", layout="wide")

# Custom CSS for Professional UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 0;
        margin-bottom: 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .main-header .subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.95;
        margin-top: 0.5rem;
    }
    
    .main-header .author {
        font-size: 0.95rem;
        font-weight: 500;
        margin-top: 1rem;
        opacity: 0.85;
        letter-spacing: 0.5px;
    }
    
    .feature-card {
        background: #f8f9fa;
        border-left: 4px solid #2a5298;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    
    .feature-card h4 {
        color: #1e3c72;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .feature-card p {
        color: #6c757d;
        font-size: 0.9rem;
        margin: 0;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #1e3c72, #2a5298);
    }
    
    .legend-box {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 2rem;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
    }
    
    .color-indicator {
        width: 20px;
        height: 20px;
        border-radius: 3px;
        margin-right: 12px;
        border: 1px solid #dee2e6;
    }
    
    .info-section {
        background: #e7f3ff;
        border-left: 4px solid #0066cc;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 6px;
        font-size: 1rem;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1a3461 0%, #234681 100%);
        box-shadow: 0 4px 12px rgba(30, 60, 114, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Professional Header
st.markdown("""
<div class="main-header">
    <h1>AUTONOMOUS DRIVING ANALYSIS SYSTEM</h1>
    <p class="subtitle">Advanced Computer Vision & Deep Learning for Road Safety Analysis</p>
    <p class="author">Developed by Suhani</p>
</div>
""", unsafe_allow_html=True)

# Model paths
LANE_MODEL = "Lane_detection/lane_detection_final_6.keras"
YOLO_TRAFFIC_WEIGHTS = "Traffic_Sign/yolov3.weights"
YOLO_TRAFFIC_CFG = "Traffic_Sign/yolov3.cfg"
YOLO_TRAFFIC_NAMES = "Traffic_Sign/coco.names"
POTHOLE_MODEL = "Pothole_detection/pothole_detector.pt"  # YOLOv8 pothole model

IMAGE_SIZE = (224, 224)
vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck in COCO

vehicle_class_names = [
    'Auto', 'Bus', 'Empty road', 'Motorcycles',
    'Tempo Traveller', 'Tractor', 'Truck', 'cars'
]

# Load all models
@st.cache_resource
def load_all_models():
    with st.spinner("Loading AI models... This may take a minute..."):
        models = {}
        
        # 1. Lane Detection Model
        try:
            models['lane'] = load_model(LANE_MODEL, custom_objects={
                'weighted_bce': weighted_bce,
                'spatial_attention': spatial_attention
            })
            st.success("[SUCCESS] Lane Detection model loaded")
        except Exception as e:
            st.warning(f"[WARNING] Lane Detection model not found: {e}")
            models['lane'] = None
        
        # 2. Traffic Sign Detection (YOLOv3)
        try:
            if os.path.exists(YOLO_TRAFFIC_WEIGHTS):
                models['traffic_net'] = cv2.dnn.readNetFromDarknet(YOLO_TRAFFIC_CFG, YOLO_TRAFFIC_WEIGHTS)
                models['traffic_labels'] = open(YOLO_TRAFFIC_NAMES).read().strip().split("\n")
                ln = models['traffic_net'].getLayerNames()
                models['traffic_ln'] = [ln[i - 1] for i in models['traffic_net'].getUnconnectedOutLayers()]
                st.success("[SUCCESS] Traffic Sign model loaded")
            else:
                models['traffic_net'] = None
                st.warning("[WARNING] Traffic Sign model not found - run traffic_sign_streamlit.py first to download")
        except Exception as e:
            st.warning(f"[WARNING] Traffic Sign model error: {e}")
            models['traffic_net'] = None
        
        # 3. Pedestrian Detection (YOLOv8)
        try:
            models['pedestrian'] = YOLO('yolov8n.pt')
            st.success("[SUCCESS] Pedestrian Detection model loaded")
        except Exception as e:
            st.warning(f"[WARNING] Pedestrian Detection model error: {e}")
            models['pedestrian'] = None
        
        # 4. Vehicle Detection (YOLOv8)
        try:
            models['vehicle_yolo'] = YOLO('yolov8n.pt')
            st.success("[SUCCESS] Vehicle Detection model loaded")
        except Exception as e:
            st.warning(f"[WARNING] Vehicle Detection model error: {e}")
            models['vehicle_yolo'] = None
        
        # Vehicle Classification (EfficientNet) - Optional
        try:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(repo_id="Coder-M/Vehicle")
            models['vehicle_classifier'] = tf.keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
            st.success("[SUCCESS] Vehicle Classification model loaded")
        except Exception as e:
            st.info("[INFO] Vehicle Classification model not available - using YOLO only")
            models['vehicle_classifier'] = None
        
        # 5. Pothole Detection (YOLOv8)
        try:
            if os.path.exists(POTHOLE_MODEL):
                models['pothole'] = YOLO(POTHOLE_MODEL)
                st.success("[SUCCESS] Pothole Detection model loaded")
            else:
                st.info("[INFO] Pothole model not found - Detection disabled. To enable: Train a model using Copy_of_POTHOLE.ipynb and save as 'Pothole_detection/pothole_detector.pt'")
                models['pothole'] = None
        except Exception as e:
            st.warning(f"[WARNING] Pothole Detection model error: {e}")
            models['pothole'] = None
        
    return models

def process_lane_detection(frame, model):
    """Detect lanes in frame with improved visibility"""
    if model is None:
        return frame
    
    original = frame.copy()
    h, w = frame.shape[:2]
    
    # Preprocess for model
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_frame = cv2.resize(input_frame, (640, 360)) / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)
    
    # Get prediction
    result = model.predict(input_frame, verbose=0)
    result = np.squeeze(result)
    result = cv2.resize(result, (w, h))
    
    # IMPROVED: Configurable threshold for better detection
    # Default 0.3, but can be adjusted via settings
    result = (result > 0.3).astype(np.uint8) * 255
    
    # Morphological operations - REDUCED for better detection
    erode_kernel = np.ones((2, 2), np.uint8)
    result = cv2.erode(result, erode_kernel, iterations=1)
    
    # IMPROVED: Smaller kernel for better lane preservation
    open_kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, open_kernel)
    
    # IMPROVED: Fewer blur iterations
    for _ in range(5):
        result = cv2.GaussianBlur(result, (7, 7), 0)
        _, result = cv2.threshold(result, 100, 255, cv2.THRESH_BINARY)
    
    # IMPROVED: Lower min area for better detection
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    result_clean = np.zeros_like(result)
    min_area = 500  # Reduced from 1500
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            result_clean[labels == label] = 255
    
    # IMPROVED: Brighter green overlay with transparency
    result_bgr = cv2.cvtColor(result_clean, cv2.COLOR_GRAY2BGR)
    result_bgr[:, :, 0] = 0  # No blue
    result_bgr[:, :, 1] = result_clean  # Full green where lanes detected
    result_bgr[:, :, 2] = 0  # No red
    
    # IMPROVED: More visible overlay (increased from 0.7 to 1.0)
    return cv2.addWeighted(original, 1, result_bgr, 1.0, 0)

def get_traffic_light_color(roi):
    """Detect traffic light color"""
    if roi.size == 0:
        return "Unknown"
    
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

def process_traffic_signs(frame, net, ln, labels):
    """Detect traffic signs and lights"""
    if net is None:
        return frame
    
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
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
            
            if labels[classID] in ["traffic light", "stop sign"] and confidence > 0.3:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            color = (0, 165, 255)  # Orange for traffic signs
            label = f"{labels[classIDs[i]]}"
            
            if labels[classIDs[i]] == "traffic light":
                roi = frame[max(0, y):min(H, y+h), max(0, x):min(W, x+w)]
                light_color = get_traffic_light_color(roi)
                label += f" ({light_color})"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def process_pedestrians(frame, model):
    """Detect pedestrians"""
    if model is None:
        return frame
    
    results = model(frame, classes=[0], conf=0.25, verbose=False)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        # Red box for pedestrians
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return frame

def process_vehicles(frame, yolo_model, classifier_model=None):
    """Detect and classify vehicles"""
    if yolo_model is None:
        return frame
    
    results = yolo_model(frame, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()
    
    for box, cls_id, conf in zip(boxes, classes, confidences):
        if cls_id not in vehicle_ids:
            continue
        
        x1, y1, x2, y2 = map(int, box[:4])
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue
        
        # Try to classify
        label = "Vehicle"
        if classifier_model is not None:
            try:
                crop_resized = cv2.resize(crop, IMAGE_SIZE)
                crop_input = preprocess_input(np.expand_dims(crop_resized.astype("float32"), axis=0))
                preds = classifier_model(crop_input, training=False)
                preds = preds["output_0"].numpy()
                label = vehicle_class_names[np.argmax(preds)]
            except:
                label = "Vehicle"
        
        # Blue box for vehicles
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame

def process_potholes(frame, model):
    """Detect potholes"""
    if model is None:
        return frame
    
    results = model(frame, conf=0.25, verbose=False)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        # Yellow box for potholes (high visibility)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(frame, f"POTHOLE {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return frame

def process_combined_frame(frame, models, lane_threshold=0.3):
    """Apply all 5 detections to a single frame"""
    # 1. Lane Detection (green overlay)
    frame = process_lane_detection(frame, models.get('lane'))
    
    # 2. Traffic Signs (orange boxes)
    if models.get('traffic_net') is not None:
        frame = process_traffic_signs(frame, models['traffic_net'], 
                                      models['traffic_ln'], models['traffic_labels'])
    
    # 3. Pedestrians (red boxes)
    frame = process_pedestrians(frame, models.get('pedestrian'))
    
    # 4. Vehicles (blue boxes)
    frame = process_vehicles(frame, models.get('vehicle_yolo'), 
                            models.get('vehicle_classifier'))
    
    # 5. Potholes (yellow boxes)
    frame = process_potholes(frame, models.get('pothole'))
    
    return frame

def process_video(video_path, models):
    """Process entire video with all detections"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error: Unable to open video file")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    frames_processed = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process with all 5 detections
        lane_thresh = models.get('lane_threshold', 0.3)
        processed_frame = process_combined_frame(frame, models, lane_thresh)
        out.write(processed_frame)
        
        frames_processed += 1
        progress = int((frames_processed / frame_count) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing: {progress}% ({frames_processed}/{frame_count} frames)")
    
    cap.release()
    out.release()
    
    progress_bar.empty()
    status_text.empty()
    
    return output_path

# Main UI Section
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="info-section">
    <h3 style="margin-top: 0; color: #1e3c72;">System Overview</h3>
    <p style="margin-bottom: 0; color: #2c3e50; font-size: 1rem;">This system integrates five advanced computer vision models to provide comprehensive road analysis in real-time.</p>
</div>
""", unsafe_allow_html=True)

# Feature Cards
st.markdown("### Detection Modules")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>Lane Detection</h4>
        <p>Neural network-based lane marking identification with spatial attention</p>
        <p style="margin-top: 8px;"><strong>Indicator:</strong> Green overlay</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>Pothole Detection</h4>
        <p>YOLOv8-based road damage detection for infrastructure monitoring</p>
        <p style="margin-top: 8px;"><strong>Indicator:</strong> Yellow boxes</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>Traffic Sign Detection</h4>
        <p>Real-time traffic light state recognition and stop sign detection</p>
        <p style="margin-top: 8px;"><strong>Indicator:</strong> Orange boxes</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>Pedestrian Detection</h4>
        <p>YOLOv8 object detection specialized for pedestrian safety</p>
        <p style="margin-top: 8px;"><strong>Indicator:</strong> Red boxes</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h4>Vehicle Classification</h4>
        <p>Multi-class vehicle identification and tracking system</p>
        <p style="margin-top: 8px;"><strong>Indicator:</strong> Blue boxes</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Settings
with st.expander("Advanced Settings"):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        lane_threshold = st.slider("Lane Detection Sensitivity", 0.1, 0.9, 0.3, 0.1,
                                   help="Lower values increase sensitivity and detect more lane markings")
    with col_s2:
        st.info("**Configuration Note:** Adjust the sensitivity threshold based on road conditions and lighting.")

st.markdown("---")

# Load models
models = load_all_models()
models['lane_threshold'] = lane_threshold  # Pass threshold to models

# File uploader section
st.markdown("### Video Upload")
st.markdown("Upload a video file to begin analysis. Supported formats: MP4, MOV, AVI, MKV")

uploaded_file = st.file_uploader("", type=['mp4', 'mov', 'avi', 'mkv'], label_visibility="collapsed")

if uploaded_file is not None:
    # Save uploaded video
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_input.write(uploaded_file.read())
    temp_input.flush()
    temp_input.close()
    
    # Display original
    st.markdown("#### Input Video")
    st.video(temp_input.name)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Start Analysis", type="primary"):
        with st.spinner("Processing video with AI models. This may take several minutes depending on video length..."):
            output_path = process_video(temp_input.name, models)
        
        if output_path:
            st.success("Processing completed successfully!")
            
            st.markdown("#### Analysis Results")
            st.video(output_path)
            
            # Download button
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="Download Processed Video",
                    data=f.read(),
                    file_name="autonomous_driving_analysis.mp4",
                    mime="video/mp4"
                )
            
            # Cleanup
            os.unlink(output_path)
    
    # Cleanup
    os.unlink(temp_input.name)

st.markdown("---")

# Professional Legend Section
st.markdown("### Detection Indicators Reference")

# Create a clean dataframe for the legend
legend_data = {
    "Module": [
        "Lane Detection",
        "Traffic Sign Detection", 
        "Pedestrian Detection",
        "Vehicle Classification",
        "Pothole Detection"
    ],
    "Indicator": [
        "Green Overlay",
        "Orange Box",
        "Red Box", 
        "Blue Box",
        "Yellow Box"
    ],
    "Description": [
        "Detected lane markings on road surface",
        "Traffic lights and stop signs with state detection",
        "Pedestrians detected in the scene",
        "Vehicles with classification (car, truck, bus, etc.)",
        "Road damage and pothole detection"
    ]
}

df = pd.DataFrame(legend_data)
st.dataframe(df, use_container_width=True, hide_index=True)

st.info("**Performance Note:** Processing time is approximately 1-2 minutes per minute of video, depending on system specifications and video resolution.")

st.markdown("<br>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #6c757d; border-top: 1px solid #dee2e6; margin-top: 2rem;">
    <p style="margin: 0;">© 2025 Autonomous Driving Analysis System | Developed by Suhani</p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Powered by TensorFlow, YOLOv8, and OpenCV</p>
</div>
""", unsafe_allow_html=True)

