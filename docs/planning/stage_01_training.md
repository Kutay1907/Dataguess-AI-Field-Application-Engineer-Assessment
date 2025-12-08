# Stage 1: Training and Model Preparation (Day 1)

**Goal**: Train a YOLOv8n model on the COCO128 dataset and prepare it for the pipeline.

## Requirements
*   **Dataset**: COCO128 (Small, ready-to-use dataset for rapid prototyping).
*   **Model Architecture**: YOLOv8n (Nano version for speed/efficiency on CPU).
*   **Framework**: `ultralytics` package.
*   **Outputs**: `models/latest.pt` (PyTorch weights).
*   **Logging**: TensorBoard or Weights & Biases (optional but recommended).

## Detailed Steps
1.  **Environment Setup**:
    *   Install `ultralytics`, `torch`, `torchvision`.
    *   Verify GPU availability (or MPS on Mac) if applicable, though training on COCO128 is fast even on CPU.

2.  **Dataset Preparation**:
    *   Download COCO128 using `ultralytics` built-in tools or manual download.
    *   Create `dataset.yaml` configuration file defining classes and paths.
    *   (Optional) Implement custom augmentations in `augmentations.py` to improve robustness.

3.  **Training Loop (`training/train.py`)**:
    *   Initialize YOLO model: `model = YOLO('yolov8n.pt')`.
    *   Run training: `model.train(data='dataset.yaml', epochs=50, imgsz=640)`.
    *   Ensure artifacts are saved to `runs/detect/`.

## Code Structure
*   `training/`
    *   `train.py`: Main training script.
    *   `dataset.yaml`: Dataset configuration.
    *   `augmentations.py`: Custom data augmentation logic.

## Verification
*   Check if `models/latest.pt` exists after training.
*   Run a quick validation on a test image to ensure the model detects objects.
