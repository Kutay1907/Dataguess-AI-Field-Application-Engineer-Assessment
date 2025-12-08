# Stage 6: Monitoring (Day 4 Afternoon)

**Goal**: Add observability to the pipeline.

## Requirements
*   **Metrics**: FPS, Inference Time, Track count.
*   **Logging**: Structured JSON logs for easier parsing.

## Detailed Steps
1.  **FPS Meter (`monitoring/fps_meter.py`)**:
    *   Window-based FPS calculation (e.g., average over last 30 frames).
    *   Class `FPSMeter`.

2.  **Structured Logger (`monitoring/logger.py`)**:
    *   Configure standard python `logging` to output JSON.
    *   Include timestamps, log levels, and context (e.g., `{"event": "detection", "count": 5}`).

3.  **Dashboard (`monitoring/dashboard.py`)**:
    *   (Dummy/Simple) A simple script or endpoint that visualizes these metrics, or just writes them to a file that can be tailed.

## Verification
*   Run the Video Engine and observe the logs.
*   Verify that FPS is calculated correctly and displayed/logged.
