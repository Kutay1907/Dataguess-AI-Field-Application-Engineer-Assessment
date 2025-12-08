# Stage 5: FastAPI + REST API (Day 4 Morning)

**Goal**: specific detection service via a RESTful API.

## Requirements
*   **Framework**: FastAPI.
*   **Persistence**: Model should be loaded *once* at startup, not per request.

## Endpoints
1.  `POST /detect`:
    *   Accepts an image file (UploadFile).
    *   Returns JSON: `{"detections": [...], "count": 5, "inference_time": 0.02}`.
2.  `GET /health`:
    *   Returns status of the service (UP/DOWN).
3.  `GET /metrics`:
    *   Returns basic usage stats (request count, avg latency).

## Implementation Details (`api/server.py`)
*   **Startup Event**: Initialize the `Detector` instance.
*   **Pydantic Models (`schemas.py`)**: Define request/response shapes for documentation and validation.
*   **Concurrency**: Use `async def` for IO-bound parts, but be careful with CPU-bound inference (might need `run_in_executor` or simple blocking if single-threaded usage is fine).

## Verification
*   Use `curl` or Postman to send an image to `localhost:8000/detect`.
*   Validate the JSON response structure.
