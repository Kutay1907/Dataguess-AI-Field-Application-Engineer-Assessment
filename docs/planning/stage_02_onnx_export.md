# Stage 2: ONNX Export (Day 2 Morning)

**Goal**: Convert the trained PyTorch model to ONNX format for optimized inference on CPU.

## Requirements
*   **Input**: `models/latest.pt`.
*   **Output**: `models/model.onnx`.
*   **Tools**: `ultralytics` export mode, `onnx`, `onnxruntime`.

## Detailed Steps
1.  **Export Script (`export_to_onnx.py`)**:
    *   Load the PyTorch model.
    *   Export to ONNX: `model.export(format='onnx', opset=12)`.
    *   Ensure dynamic axes or fixed batch size decisions are made (usually fixed for standard video pipelines, or dynamic for flexibility).

2.  **Verification & Accuracy Check**:
    *   Load both PyTorch model and ONNX model.
    *   Run inference on the same sample image.
    *   Compare outputs (bounding boxes, class scores) using `numpy.testing.assert_allclose` or IoU comparison.
    *   **Note**: TensorRT allows skipping on Mac since it relies on NVIDIA GPUs.

## Code Structure
*   `export_to_onnx.py`: Script to handle the export and verification.

## Architecture Decisions
*   **Backend**: ONNX Runtime (CPU provider) will be used for the inference engine to ensure broad compatibility and decent speed on Mac without requiring CUDA.
