# AI FAE CV Technical Assessment

## Overview
This project implements a complete Edge AI Video Analytics System capable of detecting and tracking objects in real-time using **YOLOv8** and **ByteTrack**. It features a robust inference engine, a REST API for deployment, and a monitoring system.

## Project Structure
```
cv-advanced-assessment/
├── training/       # Training pipeline & augmentations
├── inference/      # Inference engine (Detector, Tracker, Video)
├── api/            # FastAPI deployment
├── monitoring/     # FPS meter, Logger, Dashboard
├── models/         # ONNX and PyTorch models
├── tests/          # Unit and integration tests
├── report.md       # Technical Report
└── requirements.txt
```

## Setup
1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Verify Setup**
    ```bash
    pytest tests/
    ```

## Usage

### 1. Real-Time Video Engine
Run the tracker on a video file or webcam:
```bash
# Test Video (Auto-generated)
python inference/video_engine.py --source datasets/test_video.mp4

# Webcam
python inference/video_engine.py --source 0
```

### 2. Monitoring Dashboard
Open a separate terminal while the engine is running:
```bash
python monitoring/dashboard.py
```

### 3. API Service
Start the REST API:
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```
- **Documentation**: http://0.0.0.0:8000/docs
- **Health Check**: `curl http://localhost:8000/health`
- **Inference**:
  ```bash
  curl -X POST -F "file=@your_image.jpg" http://localhost:8000/detect
  ```

## Docker output
To run via Docker:
```bash
docker build -t vision-pipeline -f api/docker/Dockerfile .
docker run -p 8000:8000 vision-pipeline
```
