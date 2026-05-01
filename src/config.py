"""
Centralized configuration for the AI Parking Space Detection System.
All magic numbers, file paths, and tunable parameters live here.
"""

# --- File Paths ---
MODEL_PATH = "yolo11n.pt"
ROI_PICKLE = "Space_ROIs.pkl"
VIDEO_PATH = "input_video/parking_space.mp4"
REF_IMAGE = "ROI_Reference.png"

# --- YOLO Detection ---
CONFIDENCE_THRESHOLD = 0.35
VEHICLE_CLASSES = [2, 5, 7]  # COCO classes: 2=car, 5=bus, 7=truck

# --- Visualization ---
OVERLAY_ALPHA = 0.4
COLOR_OCCUPIED = (0, 0, 255)     # Red (BGR)
COLOR_AVAILABLE = (0, 255, 0)    # Green (BGR)
COLOR_LABEL_BG = (30, 155, 50)   # Default bounding-box label color

# --- Status Panel (OpenCV window) ---
PANEL_WIDTH = 290
PANEL_HEIGHT = 150
PANEL_BG_COLOR = (40, 40, 40)

# --- Dashboard Quality ---
QUALITY_SCALE_STANDARD = 0.6     # Downscale factor for "Standard" mode
QUALITY_SCALE_HIGH = 1.0         # Full resolution for "Professional" mode

# --- Output / Logging ---
OUTPUT_DIR = "output"
CSV_LOG_FILE = "output/occupancy_log.csv"
