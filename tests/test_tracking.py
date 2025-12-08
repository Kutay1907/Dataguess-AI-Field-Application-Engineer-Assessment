import sys
import os
import cv2
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.video_engine import VideoEngine

def create_test_video(filename="test_tracking.mp4"):
    img_path = "datasets/coco128/images/train2017/000000000009.jpg"
    height, width = 640, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
    
    if os.path.exists(img_path):
        base_img = cv2.imread(img_path)
        base_img = cv2.resize(base_img, (width, height))
    else:
        print("Warning: COCO image not found, using blank frame.")
        base_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a video by just writing the same frame (static tracking check)
    # or panning slightly
    for i in range(15):
        frame = base_img.copy()
        # Add a moving white square just to check visual updates even if detection stays same
        cv2.rectangle(frame, (i*5, 50), (i*5+50, 100), (255, 255, 255), -1)
        video.write(frame)
    video.release()
    print(f"Created {filename}")
    return filename

def test_video_engine():
    video_path = create_test_video()
    model_path = "models/model.onnx"
    
    # If model doesn't exist (e.g. CI env), create dummy file? 
    # No, we assume previous stages passed.
    if not os.path.exists(model_path):
        print("Model not found. Skipping tracking test.")
        return

    print("Initializing Video Engine...")
    engine = VideoEngine(source=video_path, model_path=model_path, backend='onnx', skip_frames=1)
    engine.start()
    
    output_frames = []
    max_frames = 50
    
    try:
        count = 0
        for frame in engine.stream_results():
            output_frames.append(frame)
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} frames", flush=True)
            # Remove early break to test full flow
    except Exception as e:
        print(f"Error during streaming: {e}")
        engine.stop()
        
    print(f"Processed {len(output_frames)} frames.")
    
    # Verify we got frames
    assert len(output_frames) > 0, "No frames processed"
    
    # Save output
    out_path = "tracked_output.mp4"
    height, width = output_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(out_path, fourcc, 20.0, (width, height))
    for f in output_frames:
        out_vid.write(f)
    out_vid.release()
    print(f"Saved {out_path}")
    
    # Clean up
    if os.path.exists(video_path):
        os.remove(video_path)

if __name__ == "__main__":
    test_video_engine()
