import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.detector import Detector
from inference.tracker import ByteTracker
from inference.fusion import FusionVisualizer

def test_sync():
    print("Sync Tracker Test")
    
    # 1. Setup
    model_path = "models/model.onnx"
    detector = Detector(model_path=model_path, backend='onnx')
    tracker = ByteTracker(track_thresh=0.4, match_thresh=0.8)
    visualizer = FusionVisualizer()
    
    # 2. Input
    # Use COCO image
    img_path = "datasets/coco128/images/train2017/000000000009.jpg"
    if not os.path.exists(img_path):
        print("COCO image missing")
        return
        
    base_img = cv2.imread(img_path)
    base_img = cv2.resize(base_img, (640, 640))
    
    # 3. Output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('sync_track.mp4', fourcc, 20.0, (640, 640))
    
    # 4. Loop
    for i in range(20):
        frame = base_img.copy()
        
        # Add visual noise/movement
        cv2.rectangle(frame, (i*10, 50), (i*10+50, 100), (255, 255, 255), -1)
        
        # Detect
        detections = detector.detect(frame)
        
        # Track
        tracks = tracker.update(detections)
        
        # Visualize
        vis = visualizer.draw_tracks(frame, tracks)
        
        out.write(vis)
        print(f"Frame {i}: {len(tracks)} tracks")
        
    out.release()
    print("Saved sync_track.mp4")

if __name__ == "__main__":
    test_sync()
