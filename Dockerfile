# --- Multi-stage Dockerfile for AI Parking Detector ---
# Stage 1: Install dependencies
FROM python:3.12-slim AS base

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application
FROM base AS app

COPY src/ src/
COPY app/ app/
COPY main.py .

# Expose Streamlit default port
EXPOSE 8501

# Default: run the Streamlit dashboard
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
