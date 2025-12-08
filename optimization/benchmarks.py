import sys
import os
import time
import numpy as np
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.detector import Detector

def benchmark_backend(backend, model_path, iterations=100):
    print(f"\nBenchmarking {backend}...")
    detector = Detector(model_path=model_path, backend=backend)
    
    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(10):
        detector.detect(dummy)
        
    start = time.perf_counter()
    for _ in range(iterations):
        detector.detect(dummy)
    end = time.perf_counter()
    
    total_time = end - start
    fps = iterations / total_time
    avg_latency = (total_time / iterations) * 1000
    
    print(f"{backend.upper()} Results ({iterations} iters):")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"FPS: {fps:.2f}")
    
    # Detailed profile
    profile = detector.get_avg_latency()
    print("Profile Breakdown (ms):", profile)
    return avg_latency, fps

if __name__ == "__main__":
    pt_path = "models/latest.pt"
    onnx_path = "models/model.onnx"
    
    if os.path.exists(pt_path):
        benchmark_backend('pytorch', pt_path)
    if os.path.exists(onnx_path):
        benchmark_backend('onnx', onnx_path)
