# Stage 8: Docker and Report (Day 5 Afternoon)

**Goal**: Containerize the application and document the architecture.

## Dockerfile
*   **Base Image**: `python:3.9-slim` or a similar lightweight image.
*   **Dependencies**: Install `libgl1-mesa-glx` (for OpenCV).
*   **Copy Code**: specific directories (`api`, `inference`, `models`, `monitoring`, `utils`).
*   **Entrypoint**: `uvicorn api.server:app --host 0.0.0.0 --port 80`.

## Report (`report.pdf` / `report.md`)
*   **Architecture Diagram**: High-level flow.
*   **Decisions**: Why ONNX? Why YOLOv8n? Why ByteTrack?
*   **Metrics**: Average FPS achieved on target hardware, Model accuracy (mAP) on validation set.
*   **Code Structure**: Brief explanation of the module organization.

## Verification
*   Build docker image: `docker build -t vision-pipeline .`.
*   Run container: `docker run -p 8000:80 vision-pipeline`.
*   Verify API health check inside the container.
