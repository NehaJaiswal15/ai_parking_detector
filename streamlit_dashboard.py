import streamlit as st
import cv2, torch, numpy as np, pickle, tempfile, os
from ultralytics import YOLO
from utilis import YOLO_Detection, drawPolygons

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Professional Parking Analytics Studio", page_icon="🅿️", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    .stMetric {text-align:center;}
    .metric-label {font-size:14px; color:#AAA;}
    .metric-value {font-size:28px; font-weight:600;}
    h1, h2, h3 {color:#9D8CFF !important;}
    </style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Professional Controls")
st.sidebar.header("Processing Settings")

label_style = st.sidebar.selectbox("Label Style", ["Clean & Minimal", "Bold & Highlighted"])
show_vehicle_ids = st.sidebar.checkbox("Show Vehicle IDs", True)
show_confidence = st.sidebar.checkbox("Show Detection Confidence", True)
output_quality = st.sidebar.selectbox("Output Quality", ["Standard", "Professional (High)"])

st.sidebar.divider()
st.sidebar.header("Status")
st.sidebar.write("Model: YOLOv11 (CPU mode)")

# ---------------- MODEL LOAD ----------------
@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = YOLO("yolo11n.pt")  # Change to yolov8n.pt if you prefer
    model.to(device)
    return model

model = load_model()

# ---------------- LOAD ROI ----------------
with open("Space_ROIs.pkl", "rb") as f:
    data = pickle.load(f)
    posList_raw = data["polygons"]
    ref_size = data["size"]

def scale_polygons(polys, ref_size, cur_size):
    ref_w, ref_h = ref_size
    cur_h, cur_w = cur_size
    sx, sy = cur_w / ref_w, cur_h / ref_h
    return [[(int(x * sx), int(y * sy)) for (x, y) in poly] for poly in polys]

# ---------------- APP HEADER ----------------
st.title("🚗 Professional Parking Analytics Studio")
st.markdown("Upload a parking lot video and analyze slot occupancy using **YOLOv8/YOLOv11**.")

uploaded_file = st.file_uploader("📤 Upload Video for Professional Processing", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    video_path = temp_file.name

    st.success("✅ Video uploaded successfully. Starting processing...")
    cap = cv2.VideoCapture(video_path)
    frame_placeholder = st.empty()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    current_frame = 0

    # --- Metrics placeholders
    col1, col2, col3, col4 = st.columns(4)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1

        cur_h, cur_w = frame.shape[:2]
        posList = scale_polygons(posList_raw, ref_size, (cur_h, cur_w))

        boxes, classes, names = YOLO_Detection(model, frame, conf=0.35)
        detection_points = [(int((x1+x2)/2), int((y1+y2)/2)) for (x1,y1,x2,y2) in boxes]

        frame, occupied_count, slot_status = drawPolygons(frame, posList, detection_points=detection_points)
        available_count = len(posList) - occupied_count
        occupancy_percent = (occupied_count / len(posList)) * 100 if len(posList) else 0

        # --- Metrics update
        col1.metric("Total Slots", len(posList))
        col2.metric("Occupied", occupied_count)
        col3.metric("Available", available_count)
        col4.metric("Occupancy (%)", f"{occupancy_percent:.1f}")

        # --- Progress
        st.progress(int((current_frame / total_frames) * 100))

        # --- Optional YOLO boxes
        if show_vehicle_ids:
            for (x1, y1, x2, y2), cls in zip(boxes, classes):
                name = names[int(cls)]
                color = (255, 255, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                label = name
                if show_confidence:
                    label += f" {model.predict(frame)[0].boxes.conf[0]:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # --- Show frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    st.success("✅ Processing completed successfully!")
else:
    st.info("📺 Upload a parking lot video to begin.")
