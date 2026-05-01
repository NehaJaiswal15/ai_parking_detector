# 🅿️ AI Parking Space Detection System

Real-time parking lot occupancy detection using **YOLOv11** and **OpenCV**, with an interactive **Streamlit** analytics dashboard. Automatically identifies occupied and available parking slots by mapping YOLO vehicle detections to user-defined polygon ROIs.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-orange?logo=yolo)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Tests](https://img.shields.io/badge/Tests-17%20Passed-brightgreen?logo=pytest)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **YOLOv11 Detection** | Detects cars, buses, and trucks using a pre-trained YOLO model |
| 🗺️ **Polygon ROI Mapping** | Interactive tool to define parking slots as 4-point polygons |
| 📊 **Real-time Analytics** | Live occupancy metrics, FPS counter, and progress tracking |
| 📈 **Trend Visualization** | Plotly-based occupancy trend chart in the dashboard |
| 📝 **CSV Logging** | Timestamped occupancy data exported for historical analysis |
| 🎛️ **Interactive Controls** | Label style, confidence toggle, and output quality settings |
| 🖥️ **Dual Interface** | OpenCV CLI mode + Streamlit web dashboard |
| ⚡ **GPU Acceleration** | Automatic CUDA detection with CPU fallback |
| 🧪 **Tested** | 17 unit tests covering core detection and logging logic |
| 🐳 **Docker Ready** | Containerized deployment with multi-stage Dockerfile |
| 🔄 **CI Pipeline** | GitHub Actions runs tests on every push |

---

## 🏗️ Architecture

```
ai_parking_detector/
├── src/                        # Core source package
│   ├── config.py               #   Centralized configuration
│   ├── detector.py             #   ParkingDetector class + DetectionResult
│   ├── utils.py                #   YOLO wrapper, polygon drawing, scaling
│   └── occupancy_logger.py     #   CSV occupancy logging
├── app/
│   └── dashboard.py            # Streamlit analytics dashboard
├── scripts/
│   └── select_spaces.py        # Interactive ROI polygon annotation tool
├── tests/
│   ├── test_utils.py           # Unit tests for utils (13 tests)
│   └── test_logger.py          # Unit tests for CSV logger (4 tests)
├── main.py                     # CLI entry point
├── Dockerfile                  # Container deployment
├── requirements.txt            # Python dependencies
└── .github/workflows/ci.yml    # GitHub Actions CI
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [YOLO model weights](https://docs.ultralytics.com/models/yolo11/) (`yolo11n.pt`)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/ai_parking_detector.git
cd ai_parking_detector

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### Step 1: Define Parking Slots (one-time)

```bash
python scripts/select_spaces.py --image ROI_Reference.png
```
- **Left-click** 4 corners to define a parking slot
- **Right-click** inside a slot to remove it
- Press **Q** to save and quit

### Step 2: Run Detection

**CLI Mode (OpenCV window):**
```bash
python main.py --video input_video/parking_space.mp4
```

**Dashboard Mode (Streamlit):**
```bash
streamlit run app/dashboard.py
```

---

## ⚙️ CLI Options

```
python main.py --help

Options:
  --video    Path to parking lot video       (default: input_video/parking_space.mp4)
  --model    Path to YOLO model weights      (default: yolo11n.pt)
  --roi      Path to ROI polygon pickle      (default: Space_ROIs.pkl)
  --conf     Detection confidence threshold  (default: 0.35)
  --no-csv   Disable CSV occupancy logging
```

---

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

```
17 passed in ~3 seconds
```

Tests cover:
- **Polygon scaling** — coordinate math across resolutions
- **Occupancy detection** — point-in-polygon logic with edge cases
- **Label rendering** — bounding box drawing and mutation verification
- **CSV logging** — file creation, headers, data integrity

---

## 🐳 Docker

```bash
docker build -t parking-detector .
docker run -p 8501:8501 parking-detector
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Object Detection | YOLOv11 (Ultralytics) |
| Computer Vision | OpenCV |
| Deep Learning | PyTorch |
| Web Dashboard | Streamlit |
| Visualization | Plotly |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Containerization | Docker |

---

## 📊 How It Works

1. **ROI Annotation** — User defines parking slot boundaries as 4-point polygons on a reference image
2. **Vehicle Detection** — YOLOv11 detects vehicles (cars, buses, trucks) in each video frame
3. **Occupancy Mapping** — For each detected vehicle, the system checks if its center point falls inside any parking slot polygon
4. **Visualization** — Slots are color-coded (🔴 occupied / 🟢 available) with real-time metrics

---

## 📄 License

This project is licensed under the MIT License.
