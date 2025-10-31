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
from ultralytics import YOLO
import tempfile

st.set_page_config(page_title="🕳️ Pothole Detection", page_icon="🕳️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #f6d365, #fda085);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🕳️ Pothole Detection System</h1>
    <p>AI-Powered Road Damage Detection using YOLOv8</p>
</div>
""", unsafe_allow_html=True)

# Information panel
with st.expander("ℹ️ About Pothole Detection"):
    st.markdown("""
    ### How it works:
    - **Model:** YOLOv8 trained on pothole dataset
    - **Detection:** Identifies potholes and road damage
    - **Output:** Yellow bounding boxes around detected potholes
    - **Confidence:** Shows detection confidence scores
    
    ### Use Cases:
    - 🚧 Road maintenance planning
    - 📊 Infrastructure assessment
    - 🚗 Autonomous vehicle navigation
    - 🛣️ Smart city applications
    """)

# Model configuration
POTHOLE_MODEL = "pothole_detector.pt"  # Trained YOLOv8 model

@st.cache_resource
def load_pothole_model():
    """Load pothole detection model"""
    with st.spinner("🔄 Loading pothole detection model..."):
        try:
            if os.path.exists(POTHOLE_MODEL):
                model = YOLO(POTHOLE_MODEL)
                st.success("✅ Pothole detection model loaded!")
                return model, True
            else:
                st.warning("⚠️ Custom pothole model not found!")
                st.info("""
                **To enable pothole detection:**
                1. Train a YOLOv8 model using the `Copy_of_POTHOLE.ipynb` notebook
                2. Save the trained model as `pothole_detector.pt` in the Pothole_detection folder
                3. Or download a pre-trained pothole detection model
                
                For now, using generic YOLOv8n model (limited accuracy for potholes).
                """)
                # Fallback to generic YOLO model
                model = YOLO('yolov8n.pt')
                return model, False
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None, False

def detect_potholes_image(image_path, model, use_custom_model, conf_threshold=0.25):
    """Detect potholes in a single image"""
    if model is None:
        return None
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        st.error("❌ Error reading image")
        return None
    
    # Run detection
    if use_custom_model:
        results = model(img, conf=conf_threshold, verbose=False)
    else:
        # Generic model - no specific pothole class
        st.warning("Using generic model - results may not be accurate for potholes")
        results = model(img, conf=conf_threshold, verbose=False)
    
    # Draw detections
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        # Get class name
        if use_custom_model:
            label = "POTHOLE"
        else:
            label = results[0].names[cls]
        
        # Yellow box for potholes
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return img

def detect_potholes_video(video_path, model, use_custom_model, conf_threshold=0.25):
    """Detect potholes in a video"""
    if model is None:
        st.error("❌ Model not loaded")
        return None
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Error opening video file")
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
    detection_count = 0
    frames_processed = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        if use_custom_model:
            results = model(frame, conf=conf_threshold, verbose=False)
        else:
            results = model(frame, conf=conf_threshold, verbose=False)
        
        # Draw detections
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            if use_custom_model:
                label = "POTHOLE"
            else:
                label = results[0].names[cls]
            
            # Yellow box for potholes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            detection_count += 1
        
        out.write(frame)
        
        frames_processed += 1
        progress = int((frames_processed / frame_count) * 100)
        progress_bar.progress(progress)
        status_text.text(f"🎬 Processing: {progress}% ({frames_processed}/{frame_count} frames) | Potholes detected: {detection_count}")
    
    cap.release()
    out.release()
    
    progress_bar.empty()
    status_text.empty()
    
    st.success(f"✅ Processing complete! Total potholes detected: {detection_count}")
    
    return output_path

# Main UI
st.markdown("---")

# Load model
model, use_custom_model = load_pothole_model()

# Settings
with st.expander("⚙️ Detection Settings"):
    conf_threshold = st.slider(
        "🎯 Confidence Threshold",
        0.1, 0.9, 0.25, 0.05,
        help="Lower = more detections (may include false positives)"
    )
    st.markdown("**Tip:** Adjust threshold based on video quality and lighting conditions")

st.markdown("---")

# Input type selection
input_type = st.radio("Select input type:", ["📹 Video", "🖼️ Image"])

if input_type == "🖼️ Image":
    st.markdown("### Upload an Image")
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file is not None:
        # Save uploaded image
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_input.write(uploaded_file.read())
        temp_input.flush()
        temp_input.close()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(temp_input.name, use_container_width=True)
        
        if st.button("🚀 Detect Potholes", type="primary"):
            with st.spinner("🔄 Analyzing image..."):
                result_img = detect_potholes_image(temp_input.name, model, use_custom_model, conf_threshold)
            
            if result_img is not None:
                # Convert BGR to RGB for display
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader("🎯 Detected Potholes")
                    st.image(result_img_rgb, use_container_width=True)
                
                # Save result for download
                result_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                cv2.imwrite(result_path, result_img)
                
                with open(result_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Result",
                        data=f.read(),
                        file_name="pothole_detection_result.jpg",
                        mime="image/jpeg"
                    )
                
                os.unlink(result_path)
        
        os.unlink(temp_input.name)

else:  # Video
    st.markdown("### Upload a Video")
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi', 'mkv'])
    
    if uploaded_file is not None:
        # Save uploaded video
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input.write(uploaded_file.read())
        temp_input.flush()
        temp_input.close()
        
        # Display original
        st.subheader("📹 Original Video")
        st.video(temp_input.name)
        
        if st.button("🚀 Detect Potholes", type="primary"):
            with st.spinner("🔄 Processing video..."):
                output_path = detect_potholes_video(temp_input.name, model, use_custom_model, conf_threshold)
            
            if output_path:
                st.subheader("🎯 Detected Potholes")
                st.video(output_path)
                
                # Download button
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Analyzed Video",
                        data=f.read(),
                        file_name="pothole_detection_result.mp4",
                        mime="video/mp4"
                    )
                
                # Cleanup
                os.unlink(output_path)
        
        # Cleanup
        os.unlink(temp_input.name)

st.markdown("---")
st.markdown("""
### 📊 Detection Info:
- **Color Code:** 🟡 Yellow boxes indicate detected potholes
- **Confidence Score:** Higher scores indicate more certain detections
- **Processing Time:** Depends on video length and resolution

### 💡 Tips for Better Results:
- Use high-quality videos with good lighting
- Ensure camera is stable for clearer detections
- Lower confidence threshold for more detections
- Train custom model on your specific road conditions

### 🔧 Model Training:
To train your own pothole detection model, use the `Copy_of_POTHOLE.ipynb` notebook included in the project.
""")

