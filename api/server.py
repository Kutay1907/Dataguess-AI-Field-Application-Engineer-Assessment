import sys
import os
import cv2
import numpy as np
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.detector import Detector
from inference.video_engine import VideoEngine
from api.schemas import DetectionResponse, Detection

# Global Detector Instance
detector = None
video_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model on startup.
    """
    # Initialize Global Video Engine
    global detector
    # We will use the detector inside the engine for single image requests too, 
    # or keep a separate one. For simplicity and resource saving, let's keep separate 
    # if we want double duty, but the prompt implies a shift to video.
    # However, existing /detect endpoint relies on global 'detector'.
    # Let's instantiate VideoEngine, which has its own detector.
    
    # We can use a test video or webcam. Let's default to a test video for "local" demo 
    # or webcam if requested. User said "local browser UI", implying maybe webcam or file.
    # Let's use '0' (webcam) as default source, or a dummy file if not available?
    # User prompt didn't specify source, but "Real-Time Video processing" usually implies camera.
    # Let's try to find a video file first to be safe (sync_track.mp4 or similar), else 0.
    
    source = "datasets/test_video.mp4"
    if not os.path.exists(source):
        print(f"Warning: {source} not found. Attempting webcam 0.")
        source = 0
        
    model_path = "models/model.onnx"
    backend = "onnx"
    if not os.path.exists(model_path):
        model_path = "models/latest.pt"
        backend = "pytorch"

    global video_engine
    print(f"Initializing VideoEngine with source={source}, model={model_path}...")
    video_engine = VideoEngine(source=source, model_path=model_path, backend=backend)
    
    # Also keep standalone detector for /detect if needed, or reuse engine's detector
    # Reusing engine's detector is cleaner to avoid 2 models in memory
    detector = video_engine.detector 
            
    yield
    if video_engine:
        video_engine.stop()
    print("Shutting down...")

app = FastAPI(title="Edge AI Detection Service", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Frontend (Optional convenience)
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
if os.path.exists(frontend_path):
    # Mount only specific files or subdirectory if needed, 
    # but mounting root is risky if API paths collide.
    # Safe strategy: Mount /app to frontend, redirect / to /app/index.html
    app.mount("/app", StaticFiles(directory=frontend_path, html=True), name="frontend")
    
    @app.get("/")
    async def root():
        return RedirectResponse(url="/app/index.html")

# Global Stats
stats = {
    "count": 0,
    "total_latency": 0.0
}

@app.get("/health")
def health_check():
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    return {"status": "UP", "backend": detector.backend}

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    if detector is None:
         raise HTTPException(status_code=503, detail="Detector not initialized")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        start_time = time.time()
        detections = detector.detect(img)
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Update Stats
        stats["count"] += 1
        stats["total_latency"] += latency

        # Convert detections to Schema
        response_dets = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            response_dets.append(Detection(
                box=[int(x1), int(y1), int(x2), int(y2)],
                confidence=float(conf),
                class_id=int(cls_id),
                label=detector.names[int(cls_id)] if hasattr(detector, 'names') and int(cls_id) < len(detector.names) else str(int(cls_id))
            ))

        return DetectionResponse(
            detections=response_dets,
            count=len(response_dets),
            inference_time=latency,
            image_size=[img.shape[1], img.shape[0]],
            backend=detector.backend
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import Response

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    if video_engine is None:
        raise HTTPException(status_code=503, detail="Video Engine not initialized")
    
    try:
        # Create uploads directory
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            # Use chunks for large files
            while content := await file.read(1024 * 1024): # 1MB chunks
                buffer.write(content)
                
        # Update engine source
        video_engine.update_source(file_path)
        
        return {"status": "success", "filename": file.filename, "message": "Video uploaded and engine restarted."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/start") # Allow GET for easier testing via browser bar if needed, but POST is proper
@app.post("/start")
def start_video():
    if video_engine is None:
        raise HTTPException(status_code=503, detail="Video Engine not initialized")
    if video_engine.capture_thread.is_alive():
        return {"status": "already_running"}
    
    video_engine.start()
    return {"status": "started"}

@app.post("/stop")
def stop_video():
    if video_engine is None:
        raise HTTPException(status_code=503, detail="Video Engine not initialized")
    video_engine.stop()
    return {"status": "stopped"}

@app.get("/frame")
def get_frame():
    if video_engine is None:
        raise HTTPException(status_code=503, detail="Video Engine not initialized")
    
    with video_engine.lock:
        if video_engine.latest_frame is None:
             # Return placeholder or 404? 
             # Let's return a 1x1 black pixel or wait? 204 No Content?
             return Response(status_code=204)
        
        frame = video_engine.latest_frame.copy()
        
    # Encode to JPEG
    ret, jpeg = cv2.imencode('.jpg', frame)
    if not ret:
        raise HTTPException(status_code=500, detail="Encoding failed")
        
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")

@app.get("/metrics")
def get_metrics():
    avg_latency = stats["total_latency"] / stats["count"] if stats["count"] > 0 else 0.0
    return {
        "request_count": stats["count"],
        "average_latency_seconds": avg_latency,
        "device": "cuda" if detector and detector.device == 'cuda' else "cpu",
        "model_loaded": detector is not None,
        "video_engine": video_engine.runtime_metrics if video_engine else None
    }
