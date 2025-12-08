import sys
import os
import torch
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.utils import letterbox, scale_boxes, non_max_suppression

def test_letterbox():
    """Test image resizing with padding."""
    img = np.zeros((100, 200, 3), dtype=np.uint8) # H=100, W=200
    new_shape = (640, 640)
    
    out, ratio, (dw, dh) = letterbox(img, new_shape, auto=False)
    
    assert out.shape == (640, 640, 3)
    # Aspect ratio check: 200/100 = 2.
    # Target: 640/640. 
    # Scale r = 640/200 = 3.2. New H = 100 * 3.2 = 320. 
    # Pading dh = (640-320)/2 = 160.
    assert ratio == (3.2, 3.2)
    assert dh == 160.0
    assert dw == 0.0

def test_scale_boxes():
    """Test standard coordinate scaling."""
    # Orig: 100x200. Scaled to 640x640 (pad h=160).
    # Box in scaled image: [0, 160, 640, 480] (Full original image area)
    boxes = torch.tensor([[0.0, 160.0, 640.0, 480.0]])
    img1_shape = (640, 640)
    img0_shape = (100, 200)
    
    scaled = scale_boxes(img1_shape, boxes.clone(), img0_shape)
    
    # Expect [0, 0, 200, 100] approximately
    expected = torch.tensor([[0.0, 0.0, 200.0, 100.0]])
    assert torch.allclose(scaled, expected, atol=1.0)

def test_nms():
    """Test Non-Max Suppression."""
    # [x1, y1, x2, y2, conf, cls]
    # Box 1: High conf
    b1 = [0, 0, 100, 100]
    # Box 2: Overlap with B1, Lower conf
    b2 = [2, 2, 98, 98] 
    # Box 3: Disjoint
    b3 = [200, 200, 300, 300]
    
    # YOLOv8 format into NMS wrapper: [Batch, Anchors, 4+Nc]
    # But my utility expects processed or raw?
    # `non_max_suppression` handles [Batch, 4+Nc, Anchors] (YOLOv8 raw)
    # Output of my detector inference is [1, 84, 8400].
    
    # Mock prediction: 3 real boxes + dummies to satisfy Anchors > Channels heuristic (Channels=5)
    p1 = [50, 50, 100, 100, 0.9] 
    p2 = [50, 50, 96, 96, 0.8] # Overlaps p1
    p3 = [250, 250, 100, 100, 0.9]
    dummy = [0, 0, 0, 0, 0.0]
    
    # Create [Batch, Channels, Anchors] = [1, 5, 10]
    preds_list = [p1, p2, p3] + [dummy]*7
    prediction = torch.tensor(preds_list).T.unsqueeze(0) # [1, 5, 10]
    
    # Run
    output = non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.45)
    
    assert len(output) == 1 # Batch size 1
    dets = output[0]
    
    # Should keep p1 and p3. p2 suppressed.
    assert len(dets) == 2
