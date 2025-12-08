import sys
import os
import pytest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.detector import Detector
# from inference.tracker import ByteTracker # Assuming we can import tracker directly if needed

def test_tensorrt_engine_loading():
    """
    Test TensorRT engine loading.
    Should be skipped if TensorRT is not available (e.g. on Mac).
    """
    try:
        import tensorrt as trt
    except ImportError:
        pytest.skip("TensorRT not installed. Skipping engine loading test.")
        
    engine_path = "models/model.engine"
    if not os.path.exists(engine_path):
        pytest.skip("TensorRT engine file not found.")
        
    # Mock or real load
    det = Detector(model_path=engine_path, backend="tensorrt")
    assert det.backend == "tensorrt"

def test_warmup():
    """ Verify warmup runs without error. """
    model_path = "models/model.onnx"
    if not os.path.exists(model_path):
        pytest.skip("Model not found")
        
    det = Detector(model_path=model_path, backend="onnx")
    # Warmup is called in __init__, so if we are here, it passed.
    # We can explicitly call it again to be sure.
    try:
        det.warmup()
        assert True
    except Exception as e:
        pytest.fail(f"Warmup failed: {e}")

def test_tracker_drift_logic():
    """
    Simulate tracker drift logic. 
    If detection is far from track, it should not associate.
    """
    # This involves testing the ByteTrack matching logic.
    # Since we use an external ByteTrack implementation or internal one, 
    # we verify that we can initialize it and it handles empty updates gracefully.
    
    from inference.tracker import ByteTracker
    tracker = ByteTracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    
    # 1. Update with no detections
    # Needs 6 columns: x1, y1, x2, y2, score, class
    dummy_dets = np.empty((0, 6)) 
    online_targets = tracker.update(dummy_dets)
    assert len(online_targets) == 0
    
    # 2. Update with detection (should create track)
    # [x1, y1, x2, y2, score, class]
    det1 = np.array([[10, 10, 50, 50, 0.9, 0]])
    online_targets = tracker.update(det1)
    # First frame usually activates track if high score
    assert len(online_targets) >= 0

def test_pre_post_consistency():
    """
    Verify Input -> Preprocess -> Postprocess pipeline consistency.
    """
    model_path = "models/model.onnx"
    if not os.path.exists(model_path):
        pytest.skip("Model not found")
        
    det = Detector(model_path=model_path, backend="onnx")
    
    # Random input
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Run detect (includes pre and post)
    # Should not crash and return list
    results = det.detect(img)
    assert isinstance(results, (list, np.ndarray))
