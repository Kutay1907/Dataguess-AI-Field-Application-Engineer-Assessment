import requests
import time
import sys
import os

def test_api():
    base_url = "http://localhost:8000"
    
    # 1. Health Check
    print("Checking /health...")
    try:
        resp = requests.get(f"{base_url}/health")
        if resp.status_code != 200:
            print(f"Health check failed: {resp.status_code} {resp.text}")
            sys.exit(1)
        print("Health check passed.")
    except Exception as e:
        print(f"Could not connect to server: {e}")
        sys.exit(1)

    # 2. Detect
    print("Checking /detect...")
    img_path = "inference_test.jpg"
    if not os.path.exists(img_path):
        print(f"Test image {img_path} not found.")
        sys.exit(1)
        
    with open(img_path, "rb") as f:
        files = {"file": f}
        resp = requests.post(f"{base_url}/detect", files=files)
        
    if resp.status_code != 200:
        print(f"Detect failed: {resp.status_code} {resp.text}")
        sys.exit(1)
        
    data = resp.json()
    print("Detection response:", data)
    
    if "detections" not in data:
         print("Invalid response format.")
         sys.exit(1)
         
    print("API Verification Passed!")

if __name__ == "__main__":
    # Wait for server to start
    time.sleep(2)
    test_api()
