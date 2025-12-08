# Stage: Frontend Development for Local CV Demo

## Objective
Implement a minimal, production-style local frontend that allows users to:
*   Upload an image
*   Send it to the FastAPI backend (POST /detect)
*   Display detection results visually (bounding boxes, labels)
*   Show inference metrics (latency, backend type, detection count)

The frontend must run locally, without any external dependencies or frameworks.

## Requirements

### 1. Technology Stack
Use only:
*   HTML
*   CSS
*   Vanilla JavaScript
*   Canvas API

No React, no Streamlit, no build tools.
Browser must open index.html directly (no server required).

### 2. Deliverables

#### 2.1 File Structure
```
frontend/
│── index.html
│── script.js
│── draw.js
│── style.css
```
All files must be self-contained and easy to run.

#### 2.2 Features

**A) Image Upload**
*   User selects image via `<input type="file">`
*   Image is displayed on `<canvas>`

**B) Backend Request**
*   Send POST request:
    *   `POST http://127.0.0.1:8000/detect`
    *   `Content-Type: multipart/form-data`
    *   Payload: `file=<uploaded_image>`
*   Response JSON example:
    ```json
    {
      "detections": [
        {
          "bbox": [x1, y1, x2, y2],
          "score": 0.92,
          "class_name": "person"
        }
      ],
      "latency_ms": 35.1,
      "backend": "onnx"
    }
    ```

**C) Rendering**
*   Redraw original image on canvas
*   Draw bounding boxes and labels
*   Use green stroke and white text
*   Label format: `class_name (score%)`

**D) Metrics Panel**
*   Display:
    *   Latency (ms)
    *   Backend type (onnx/pytorch)
    *   Number of detections

## 3. Acceptance Criteria
The frontend is considered complete when:
*   index.html opens locally by double-click
*   User uploads an image
*   Backend returns detection results
*   Canvas displays bounding boxes correctly
*   Latency/backend/detection count are visible
*   No console errors
*   Code is short, modular, and readable

## 4. Notes for Coding Agent
*   Keep JS modular: one file for API logic (script.js), one for drawing (draw.js)
*   Avoid any library or bundler
*   Make sure canvas resizes to match uploaded image
*   Use fetch() with FormData for upload
*   Ensure cross-origin issues do not occur (backend must allow CORS)
*   Write clean, commented code

## 5. Optional Enhancements (Not required)
*   Drag & drop image upload
*   Color-coded bounding boxes by class
*   Dark mode toggle
*   Local FPS counter
*   Video upload support
