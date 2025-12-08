# Stage 3: Inference Engine (Day 2 Afternoon)

**Goal**: Build a unified `Detector` class that abstracts the underlying inference backend (PyTorch vs. ONNX).

## Requirements
*   **Class**: `Detector`.
*   **Inputs**: Frame (numpy array).
*   **Outputs**: Standardized detections list/array (e.g., `[[x1, y1, x2, y2, score, class_id], ...]`).

## Detailed Steps
1.  **Detector Design (`inference/detector.py`)**:
    *   Initialize with config (model path, backend type, confidence threshold, NMS threshold).
    *   Implement `preprocess(image)`: Resize, normalize, transmit layout (HWC -> CHW).
    *   Implement `postprocess(output)`: Decode raw model outputs, apply Non-Maximum Suppression (NMS), scale boxes back to original image size.

2.  **Backend Implementation**:
    *   **PyTorch**: Uses `ultralytics` `model.predict()` or raw torch inference.
    *   **ONNX**: Uses `onnxruntime.InferenceSession`.

3.  **Utils (`utils.py`)**:
    *   Helper functions for visualization (drawing boxes).
    *   NMS implementation (if doing raw ONNX inference logic manually).

## Key Features
*   **Abstraction**: The rest of the codebase shouldn't care if the model is ONNX or PyTorch.
*   **Performance**: Minimize numpy copies.

## Verification
*   Run the detector on a video file and display the output.
