import os
import sys
import cv2
import pytest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.detector import Detector

def test_stage_1_training_artifacts():
    """Verify Stage 1: Model Training artifacts exist."""
    assert os.path.exists("models/latest.pt"), "Stage 1 Failed: models/latest.pt not found."
    # Optional: check size > 0
    assert os.path.getsize("models/latest.pt") > 1000, "Model file seems too small."

def test_stage_2_onnx_artifacts():
    """Verify Stage 2: ONNX Export artifacts exist."""
    assert os.path.exists("models/model.onnx"), "Stage 2 Failed: models/model.onnx not found."

def test_stage_3_inference():
    """Verify Stage 3: Inference Engine works and detects objects."""
    model_path = "models/model.onnx"
    img_path = "datasets/coco128/images/train2017/000000000009.jpg"
    
    assert os.path.exists(model_path), "Model missing for inference test."
    assert os.path.exists(img_path), "Test image missing."
    
    # Initialize Detector (ONNX Backend)
    detector = Detector(model_path=model_path, backend='onnx', conf_thres=0.25)
    
    # Load Image
    img = cv2.imread(img_path)
    assert img is not None, "Failed to load test image."
    
    # Run Detection
    detections = detector.detect(img)
    
    # Verify Detections
    # with 20 epoch training, we expect some food items to be detected
    print(f"\nFound {len(detections)} detections.")
    assert len(detections) > 0, "Stage 3 Failed: No detections found on test image."
    
    # Check format of detections: [x1, y1, x2, y2, conf, cls]
    if len(detections) > 0:
        det = detections[0]
        assert det.shape == (6,), f"Detection shape incorrect: {det.shape}, expected (6,)"
        assert det[4] >= 0.25, "Confidence below threshold in output."

if __name__ == "__main__":
    # Manually run if executed as script
    try:
        test_stage_1_training_artifacts()
        print("✅ Stage 1 Check Passed")
        test_stage_2_onnx_artifacts()
        print("✅ Stage 2 Check Passed")
        test_stage_3_inference()
        print("✅ Stage 3 Check Passed")
        print("\n🎉 ALL STAGES (1-3) PASSED VERIFICATION.")
    except AssertionError as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
