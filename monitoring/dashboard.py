import time
import json
import os
import sys

def tail_dashboard(log_file="logs/system.log"):
    """
    Tails the log file and prints key metrics in a 'dashboard' style in the terminal.
    """
    print(f"Starting Dashboard. Monitoring {log_file}...")
    print("Press Ctrl+C to exit.")
    
    if not os.path.exists(log_file):
        print(f"Waiting for log file {log_file} to be created...")
        while not os.path.exists(log_file):
            time.sleep(1)
            
    try:
        with open(log_file, "r") as f:
            # Go to the end of file
            f.seek(0, os.SEEK_END)
            
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                
                try:
                    data = json.loads(line)
                    # Simple visualization logic
                    timestamp = data.get("timestamp", "")
                    level = data.get("level", "INFO")
                    event = data.get("event", "")
                    
                    if event == "inference":
                        fps = data.get("fps", 0.0)
                        latency = data.get("latency", 0.0)
                        count = data.get("count", 0)
                        
                        # Print status line
                        # Use carriage return to overwrite line if possible, or just print new lines for log stream
                        print(f"[{timestamp}] FPS: {fps:.2f} | Latency: {latency:.4f}s | Objects: {count}")
                        
                    elif level == "ERROR":
                         print(f"🔴 ERROR: {data.get('message')}")
                         
                except json.JSONDecodeError:
                    pass
                    
    except KeyboardInterrupt:
        print("\nStopping Dashboard.")

if __name__ == "__main__":
    log_path = "logs/system.log"
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    tail_dashboard(log_path)
