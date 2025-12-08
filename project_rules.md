# AI FAE CV Technical Assessment - Project Rules

## 1. Project Overview & Deliverables
*   **Goal**: Build a fully working Edge AI Video Analytics System.
*   **Deadline**: 12 December 2025 – 18:00 (GMT+3).
*   **Deliverables**:
    *   GitHub Repository (Full project).
    *   PDF Technical Report.
    *   (Optional) Demo Video.

## 2. Recommended Directory Structure
The project must strictly follow this structure:
```
cv-advanced-assessment/
├── training/ (train.py, dataset.yaml, augmentations.py)
├── optimization/ (export_to_onnx.py, build_trt_engine.py, calibrate_int8.py, benchmarks.py)
├── inference/ (detector.py, tracker.py, video_engine.py, fusion.py, utils.py)
├── api/ (server.py, schemas.py, docker/Dockerfile)
├── monitoring/ (logger.py, fps_meter.py, dashboard.py)
├── tests/ (test_inference.py, test_onnx_shapes.py, test_tracker.py)
├── models/ (latest.pt, model.onnx, model_fp16.engine, model_int8.engine, calibration.cache)
├── README.md
└── report.pdf
```

## 3. Implementation Requirements

### Part 1: Model Training
*   **Model**: YOLOv8/v11, YOLOv7, Detectron2, EfficientDet, RT-DETR, or DEIMv2.
*   **Augmentation**: **Strong** usage of Albumentations (Mosaic, MixUp, CutOut, MotionBlur, RandomCrop, ColorJitter).
*   **Training Config**:
    *   Multi-scale training.
    *   EMA (Exponential Moving Average).
    *   Cosine LR schedule.
    *   AMP (Mixed Precision).
*   **Logging**: Loss curves, mAP@0.5, mAP@0.5:0.95, Confusion Matrix (TensorBoard or W&B).

### Part 2: Model Optimization Pipeline
*   **PyTorch to ONNX**:
    *   Dynamic axes (Batch size, Height, Width).
    *   Opset version ≥ 12.
    *   Validation step: ONNX outputs must match PyTorch.
*   **TensorRT Generation**:
    *   Generate **FP16** and **INT8** engines.
    *   Using `build_trt_engine.py`.
    *   Optimization profiles: Min/Opt/Max resolutions.
    *   Workspace size tuning.
*   **INT8 Calibration**:
    *   Entropy-based calibration using a dataset of 200-500 images.
    *   Save cache to `/models/calibration.cache`.
*   **Benchmarking**:
    *   Metrics: Latency (avg, p50, p95), Throughput (FPS), GPU/CPU stats.
    *   Warmup: ≥ 10 iterations.

### Part 3: Multi-Backend Inference Engine
*   **Class**: `Detector`
*   **Backends**:
    1.  PyTorch
    2.  ONNX Runtime (CPU / CUDA / TensorRT EP) - *Note: TensorRT EP on Linux/NVIDIA, CPU on Mac.*
    3.  Native TensorRT (Python API)
*   **Features**:
    *   Consistent Pre/Post-processing across all backends.
    *   Custom NMS.
    *   Batch inference support.
    *   Warm-up logic.
    *   Timing statistics.

### Part 4: Real-Time Video Engine
*   **Hybrid Architecture**: Video -> Detector -> Tracker -> Fusion -> Vis.
*   **Tracking**: ByteTrack, DeepSORT, or OpenCV CSRT/KCF.
*   **Logic**:
    *   Run Detector every N frames.
    *   Track in between.
    *   **Drift Detection**: If IoU(detection, track) < 0.5 -> Reinitialize tracker.
*   **Threading**: Minimum 2 threads (Detector, Visualization).

### Part 5: API Deployment
*   **Framework**: FastAPI.
*   **Endpoints**:
    *   `/detect`: Returns valid JSON (bbox, confidence, time).
    *   `/health`: Service status.
    *   `/metrics`: Latency, FPS, GPU usage.
*   **Docker**:
    *   Base image: NVIDIA TensorRT Runtime.
    *   Features: Auto-load engine, GPU-enabled.

### Part 6: Performance Monitoring
*   **Modules**: `fps_meter.py`, `logger.py`.
*   **Metrics**: FPS, GPU Mem/Util, Latency Histogram (p50/p90/p95).
*   **Logging**: JSON format.

### Part 7: Unit Tests
*   **Framework**: `pytest`.
*   **Coverage**:
    *   ONNX dynamic shapes.
    *   TensorRT engine loading.
    *   Tracker drift logic.
    *   Pre/Post-process consistency.
    *   I/O shape validation.

## 4. Coding Standards & Best Practices

### Key Principles
*   Write **clean, efficient, and well-documented code**.
*   Follow **PEP 8** style guidelines.
*   Use **type hints** for better code clarity.
*   Implement proper **error handling**.
*   Write **modular and reusable** code.

### SOLID Design Principles
*   **S**ingle Responsibility Principle: A class/function should have one and only one reason to change.
*   **O**pen/Closed Principle: Software entities should be open for extension but closed for modification.
*   **L**iskov Substitution Principle: Subtypes must be substitutable for their base types.
*   **I**nterface Segregation Principle: Many client-specific interfaces are better than one general-purpose interface.
*   **D**ependency Inversion Principle: Depend upon abstractions, not concretions.

### Python Best Practices
*   Use **virtual environments** (venv, conda).
*   Use `requirements.txt` or `pyproject.toml` for dependencies.
*   Follow naming conventions (snake_case for functions/variables).
*   Use **list comprehensions** and generator expressions.
*   Use **context managers** (`with` statement).
*   Implement proper **logging**.

### Machine Learning & Deep Learning
*   **Libraries**:
    *   `scikit-learn` for traditional ML.
    *   **PyTorch** or TensorFlow for deep learning.
*   **Workflow**:
    *   Implement proper **data preprocessing** and **validation**.
    *   Use **cross-validation** for evaluation.
    *   Track experiments with **MLflow** or **Weights & Biases** (W&B).
    *   **Version control** datasets and models.
    *   Use **data augmentation**.
    *   Implement **early stopping** and **checkpointing**.
    *   Use **GPU acceleration** when available.
    *   Monitor training with **TensorBoard**.

### Data Processing
*   Use **pandas** for data manipulation.
*   Use **numpy** for numerical computations.
*   Use **matplotlib/seaborn** for visualization.
*   Handle **missing data** appropriately.
*   Use **efficient data structures**.

### Model Deployment (FastAPI focus)
*   **Framework**: **FastAPI** (Project Standard) or Flask.
*   **Best Practices**:
    *   Implement **model versioning**.
    *   Use **Docker** for containerization.
    *   Implement proper **API documentation**.
    *   Add **input validation** and **error handling**.
    *   Monitor model performance in production.

### Testing
*   Write **unit tests** with `pytest`.
*   Test **data pipelines** and **model predictions**.
*   Use **fixtures** for test data.
*   Implement **integration tests**.

### Performance Optimization
*   Use **vectorization** with `numpy`.
*   Use **multiprocessing** for CPU-bound tasks.
*   Use **async/await** for I/O-bound tasks.
*   Profile code to identify bottlenecks.
*   Use **Cython** or **numba** for optimization if necessary.
