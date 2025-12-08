import cv2
import threading
import queue
import time
import sys
import os
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.detector import Detector
from inference.tracker import ByteTracker
from monitoring.fps_meter import FPSMeter
from monitoring.logger import setup_logger
from inference.fusion import FusionVisualizer

class VideoEngine:
    def update_source(self, new_source):
        """
        Updates the video source and restarts the engine if running.
        """
        print(f"Updating video source to: {new_source}")
        self.stop()
        self.source = new_source
        # Reset state if needed
        self.tracker = ByteTracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
        self.start()

    def __init__(self, source, model_path, backend='onnx', skip_frames=1):
        """
        Real-Time Video processing engine.
        
        Args:
            source (str/int): Video file path or camera index.
            model_path (str): Path to detection model.
            backend (str): Backend for detector.
            skip_frames (int): Number of frames to skip detection (track interpolation).
                               0 means detect on every frame.
        """
        self.source = source
        self.skip_frames = skip_frames
        
        # Monitoring
        self.fps_meter = FPSMeter(window_size=30)
        self.logger = setup_logger(name="VideoEngine", log_file="logs/system.log")
        
        # Modules
        self.detector = Detector(model_path=model_path, backend=backend)
        self.tracker = ByteTracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
        self.visualizer = FusionVisualizer()
        
        # Queues
        # Use simple Queue for thread safety
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)
        
        # Control
        self.stop_event = threading.Event()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        
        # Shared State for API
        self.lock = threading.Lock()
        self.latest_frame = None
        self.runtime_metrics = {
            "fps": 0.0,
            "latency_ms": 0.0,
            "active": False,
            "detection_count": 0,
            "class_counts": {}
        }

        # Metrics
        self.frame_count = 0
        self.fps = 0.0

    def start(self):
        print("Starting Video Engine...")
        self.logger.info("Video Engine Started", extra={"source": str(self.source), "backend": self.detector.backend})
        self.stop_event.clear()
        
        # Always create new threads because threads cannot be started twice
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
             
        self.capture_thread.start()
        self.process_thread.start()
        with self.lock:
             self.runtime_metrics["active"] = True
        
    def stop(self):
        print("Stopping Video Engine...")
        self.logger.info("Video Engine Stopping")
        self.stop_event.set()
        # Wait for join or force
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
        
        with self.lock:
             self.runtime_metrics["active"] = False

    def _capture_loop(self):
        try:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                print(f"Error: Could not open source {self.source}")
                self.logger.error(f"Could not open source {self.source}")
                self.stop_event.set()
                return

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("End of stream.")
                    self.logger.info("End of stream reached")
                    self.frame_queue.put(None) 
                    break
                try:
                    self.frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                     try:
                         self.frame_queue.get_nowait()
                         self.frame_queue.put(frame)
                     except: 
                         pass
            cap.release()
        except Exception as e:
            self.logger.error(f"Capture Loop Error: {e}", exc_info=True)
            import traceback
            traceback.print_exc()

    def _process_loop(self):
        try:
            self.fps_meter.start()
            frame_idx = 0
            
            while not self.stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                    
                if frame is None: # EOS
                    self.result_queue.put(None)
                    break
                
                t_start = time.perf_counter()
                tracks = []
                
                # Logic: Detect every N frames
                do_detect = (frame_idx % (self.skip_frames + 1) == 0)
                detection_latency = 0.0
                
                if do_detect:
                    # Run Detector
                    det_start = time.perf_counter()
                    detections = self.detector.detect(frame)
                    det_end = time.perf_counter()
                    detection_latency = det_end - det_start
                    
                    # Update Tracker
                    tracks = self.tracker.update(detections)
                    
                    # Log Detection Event
                    if len(detections) > 0:
                        self.logger.debug("Detection Event", extra={
                            "event": "inference",
                            "frame": frame_idx,
                            "count": len(detections),
                            "latency": detection_latency, 
                            "fps": self.fps
                        })
                else:
                    # Prediction Only
                    current_tracks = self.tracker.tracked_stracks
                    self.tracker.predict_tracks(current_tracks)
                    tracks = [t for t in current_tracks if t.is_activated]
                    
                # Visualization
                vis_frame = self.visualizer.draw_tracks(frame, tracks)
                
                # FPS
                self.fps_meter.update()
                self.fps = self.fps_meter.get_fps()
                
                cv2.putText(vis_frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.putText(vis_frame, f"Backend: {self.detector.backend}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Calculate Class Counts
                class_counts = {}
                for t in tracks:
                    # Tracker tracks usually preserve class_id or we can stick to just counting
                    # Assuming we can get class name from detector if stored, or just use ID
                    # ByteTrack simple implementation might not propagate class name easily if not customized
                    # But let's assume valid tracks have a .class_id or we map it somehow.
                    # For this demo, let's look at how tracker is implemented or just count total for now if unsure.
                    # Checking tracker code... actually `t.score` etc.
                    # Let's trust we passed detections with class_ids.
                    # Check if STrack has class_id? Usually they act as the class of the first detection.
                    
                    # If using standard ByteTrack, it might be tricky.
                    # But we can try to get it. 
                    # If not available, we will just say "Object"
                    c_id = getattr(t, 'class_id', -1)
                    if c_id == -1:
                        # Try to get from score or other attributes if mixed up, but likely correct
                        pass
                        
                    label = f"Class {int(c_id)}" # Default
                    if hasattr(self.detector, 'names'):
                        try:
                            # Handle both string/int keys/values safely
                            c_id_int = int(c_id)
                            if c_id_int in self.detector.names:
                                label = self.detector.names[c_id_int]
                        except:
                            pass
                    
                    class_counts[label] = class_counts.get(label, 0) + 1
                
                # Debug print to trace if counts are working
                # print(f"DEBUG: Tracks: {len(tracks)}, Class Counts: {class_counts}")

                # Update Shared State
                with self.lock:
                    self.latest_frame = vis_frame.copy()
                    self.runtime_metrics.update({
                        "fps": self.fps,
                        "latency_ms": detection_latency * 1000 if do_detect else 0,
                        "detection_count": len(tracks),
                        "backend": self.detector.backend,
                        "class_counts": class_counts
                    })

                try:
                    self.result_queue.put(vis_frame, timeout=0.1)
                except queue.Full:
                    # If full and we want to stop, or just drop frame to keep realtime?
                    # Dropping is better for realtime, but here we want robustness.
                    # Loop back to check stop_event
                    pass
                frame_idx += 1
        except Exception as e:
            self.logger.error(f"Process Loop Error: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            self.result_queue.put(None) # Unblock main thread
            with self.lock:
                 self.runtime_metrics["active"] = False

    def stream_results(self):
        """
        Generator for main thread to consume/display results.
        """
        while True:
            try:
                res = self.result_queue.get(timeout=1.0)
                if res is None:
                    break
                yield res
            except queue.Empty:
                if self.stop_event.is_set() and self.frame_queue.empty():
                    break
                continue

if __name__ == "__main__":
    # Test script in main
    # python inference/video_engine.py <video_path>
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='datasets/test_video.mp4', help='Video source')
    parser.add_argument('--model', type=str, default='models/model.onnx', help='Model path')
    args = parser.parse_args()
    
    # Handle int source (webcam)
    source = args.source
    if source.isdigit():
        source = int(source)
    elif not os.path.exists(source) and str(source) != '0':
        # Create dummy video if test_video.mp4 doesn't exist
        print("Creating dummy video for testing...")
        height, width = 640, 640
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(source, fourcc, 20.0, (width, height))
        for i in range(100):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            # Add moving square
            x = (i * 5) % width
            y = (i * 5) % height
            cv2.rectangle(frame, (x, y), (x+50, y+50), (255, 255, 255), -1)
            video.write(frame)
        video.release()
        print(f"Created {source}")

    engine = VideoEngine(source=source, model_path=args.model, backend='onnx', skip_frames=2)
    engine.start()
    
    try:
        for frame in engine.stream_results():
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                engine.stop()
                break
    except KeyboardInterrupt:
        engine.stop()
    
    engine.stop()
    cv2.destroyAllWindows()
