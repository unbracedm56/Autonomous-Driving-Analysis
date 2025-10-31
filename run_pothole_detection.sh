#!/bin/bash

# Navigate to parent directory and activate virtual environment
cd "$(dirname "$0")/.."
source venv/bin/activate

# Set threading environment variables
export OMP_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Run Pothole Detection App
streamlit run Pothole_detection/pothole_detection_streamlit.py

