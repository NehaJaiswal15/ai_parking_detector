"""
AI Parking Analytics Studio — Streamlit Dashboard.
Upload a parking-lot video to analyze slot occupancy in real time
using YOLOv11 with interactive controls, occupancy trend chart,
and performance metrics.
"""

import logging
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

# Ensure project root is on the path so 'src' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    MODEL_PATH, ROI_PICKLE,
    QUALITY_SCALE_STANDARD, QUALITY_SCALE_HIGH,
)
from src.detector import ParkingDetector
from src.occupancy_logger import OccupancyLogger

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-18s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Parking Analytics Studio", page_icon="🅿️", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    .stMetric {text-align:center;}
    h1, h2, h3 {color:#9D8CFF !important;}
    </style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Control Panel")
st.sidebar.header("Settings")

label_style = st.sidebar.selectbox("Label Style", ["Clean & Minimal", "Bold & Highlighted"])
show_vehicle_ids = st.sidebar.checkbox("Show Vehicle IDs", True)
show_confidence = st.sidebar.checkbox("Show Detection Confidence", True)
output_quality = st.sidebar.selectbox("Output Quality", ["Standard", "Professional (High)"])

QUALITY_SCALE = QUALITY_SCALE_HIGH if output_quality == "Professional (High)" else QUALITY_SCALE_STANDARD

st.sidebar.divider()
st.sidebar.header("Status")
device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
st.sidebar.write(f"Model: YOLOv11 ({device_name})")

# ---------------- DETECTOR (cached) ----------------
@st.cache_resource
def get_detector() -> ParkingDetector:
    """Load and cache the ParkingDetector instance."""
    return ParkingDetector(model_path=MODEL_PATH, roi_path=ROI_PICKLE)

detector = get_detector()

# ---------------- APP HEADER ----------------
st.title("🚗 AI Parking Analytics Studio")
st.markdown("Upload a parking-lot video to analyze slot occupancy using **YOLOv11** and visualize results below.")

uploaded_file = st.file_uploader("📤 Upload a Parking Lot Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    video_path = temp_file.name

    st.success("✅ Video uploaded successfully. Starting processing...")
    logger.info("Processing uploaded video: %s", uploaded_file.name)

    cap = cv2.VideoCapture(video_path)
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    current_frame = 0

    # --- Metrics row ---
    col1, col2, col3, col4, col5 = st.columns(5)

    # --- Trend chart placeholder ---
    st.subheader("📈 Occupancy Trend")
    chart_placeholder = st.empty()

    # --- Data collection for trend chart ---
    frame_numbers: list[int] = []
    occupancy_history: list[float] = []

    # --- CSV Logger ---
    csv_logger = OccupancyLogger()

    # --- FPS tracking ---
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1

        # --- FPS calculation ---
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
        prev_time = current_time

        # --- Apply output quality scaling ---
        if QUALITY_SCALE != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * QUALITY_SCALE), int(h * QUALITY_SCALE)))

        # --- Run detection ---
        result = detector.detect_frame(frame)
        frame = result.frame

        # --- Log to CSV ---
        csv_logger.log(
            frame_num=current_frame,
            total_slots=result.total_slots,
            occupied=result.occupied,
            available=result.available,
            occupancy_pct=result.occupancy_percent,
        )

        # --- Collect trend data ---
        frame_numbers.append(current_frame)
        occupancy_history.append(result.occupancy_percent)

        # --- Update metrics ---
        col1.metric("Total Slots", result.total_slots)
        col2.metric("Occupied", result.occupied)
        col3.metric("Available", result.available)
        col4.metric("Occupancy (%)", f"{result.occupancy_percent:.1f}")
        col5.metric("FPS", f"{fps:.1f}")

        progress_bar.progress(int((current_frame / total_frames) * 100))

        # --- Draw vehicle bounding boxes ---
        if show_vehicle_ids:
            for (x1, y1, x2, y2), cls, cf in zip(result.boxes, result.classes, result.confidences):
                name = result.names[int(cls)]
                color = (255, 255, 255)

                if label_style == "Bold & Highlighted":
                    thickness = 2
                    font_scale = 0.7
                    label_text = f"{name} {cf:.2f}" if show_confidence else name
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    cv2.rectangle(frame, (int(x1), int(y1) - th - 8), (int(x1) + tw + 6, int(y1)), (0, 0, 0), -1)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), thickness)
                    cv2.putText(frame, label_text, (int(x1) + 3, int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 200, 255), thickness)
                else:
                    thickness = 1
                    font_scale = 0.5
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    label_text = f"{name} {cf:.2f}" if show_confidence else name
                    cv2.putText(frame, label_text, (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # --- Update trend chart every 10 frames (performance optimization) ---
        if current_frame % 10 == 0 or current_frame == total_frames:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=frame_numbers,
                y=occupancy_history,
                mode="lines",
                name="Occupancy %",
                line=dict(color="#9D8CFF", width=2),
                fill="tozeroy",
                fillcolor="rgba(157, 140, 255, 0.15)",
            ))
            fig.update_layout(
                xaxis_title="Frame",
                yaxis_title="Occupancy (%)",
                yaxis=dict(range=[0, 100]),
                template="plotly_dark",
                height=300,
                margin=dict(l=40, r=20, t=10, b=40),
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)

    cap.release()
    logger.info("Dashboard processing complete. %d frames.", current_frame)
    st.success("✅ Processing completed successfully!")

    # --- Final trend chart ---
    if frame_numbers:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frame_numbers,
            y=occupancy_history,
            mode="lines",
            name="Occupancy %",
            line=dict(color="#9D8CFF", width=2),
            fill="tozeroy",
            fillcolor="rgba(157, 140, 255, 0.15)",
        ))
        fig.update_layout(
            xaxis_title="Frame",
            yaxis_title="Occupancy (%)",
            yaxis=dict(range=[0, 100]),
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=20, t=10, b=40),
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

else:
    st.info("📺 Upload a parking-lot video to begin analysis.")