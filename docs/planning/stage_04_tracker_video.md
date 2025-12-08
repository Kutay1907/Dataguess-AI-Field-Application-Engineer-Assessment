# Stage 4: Tracker and Video Engine (Day 3)

**Goal**: Integrate object tracking and build a multi-threaded video processing engine.

## Requirements
*   **Tracker**: ByteTrack (Robust multi-object tracking).
*   **Logic**: Run detection every N frames (e.g., every 5th frame) to save compute; use tracker to interpolate in between.
*   **Drift Control**: Re-initialize tracking if Intersection over Union (IoU) drops below 0.5 significantly.

## Detailed Steps
1.  **Tracker Integration (`inference/tracker.py`)**:
    *   Wrapper around ByteTrack implementation.
    *   Input: Detections from the `Detector`.
    *   Output: Track IDs for each detection.

2.  **Fusion Logic (`fusion.py`)**:
    *   Matches current detections with existing tracks.
    *   Handles ID persistency.

3.  **Video Engine (`video_engine.py`)**:
    *   **Threading**:
        *   `CaptureThread`: Reads frames from video source/webcam.
        *   `ProcessingThread`: Runs Detection + Tracking.
        *   `Display/OutputThread`: Renders results or pushes to stream.
    *   **Optimization**: Skip frames if processing is lagging.

## Code Structure
*   `video_engine.py`: Main driver class.
*   `inference/tracker.py`: Tracking logic.
*   `fusion.py`: Merging detection and tracking data.

## Verification
*   Process a test video.
*   Verify that object IDs remain stable even if detection is skipped for a few frames.
*   Test robustness against occlusions.
