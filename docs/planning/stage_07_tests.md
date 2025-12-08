# Stage 7: Tests (Day 5 Morning)

**Goal**: Ensure reliability and correctness of the pipeline components.

## Scope
*   Unit tests for critical components.
*   Integration tests for the full pipeline.

## Test Cases (`tests/`)
1.  `test_inference.py`:
    *   Compare PyTorch and ONNX outputs again (CI/CD style).
    *   Ensure NMS works as expected (removes duplicates).
2.  `test_tracker.py`:
    *   Feed synthetic detections and verify ID generation logic.
3.  `test_onnx_shapes.py`:
    *   Verify input shape requirements (batch size, channel ordering).

## Tools
*   `pytest` framework.

## Verification
*   Run `pytest` and achieve green status on all tests.
