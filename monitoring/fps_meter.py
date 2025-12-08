import time
import collections

class FPSMeter:
    """
    Tracks and calculates Frames Per Second (FPS) using a moving average window.
    """
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.timestamps = collections.deque(maxlen=window_size)
        self.start_time = None
        
    def start(self):
        """Reset the timer."""
        self.start_time = time.time()
        self.timestamps.clear()
        
    def update(self):
        """Record a new frame timestamp."""
        now = time.time()
        self.timestamps.append(now)
        
    def get_fps(self):
        """Calculate current FPS based on the window."""
        if len(self.timestamps) < 2:
            return 0.0
            
        # Calculate time difference between oldest and newest frame in window
        duration = self.timestamps[-1] - self.timestamps[0]
        if duration <= 0:
            return 0.0
            
        # Frames processed in that duration is length - 1 (intervals)
        count = len(self.timestamps) - 1
        return count / duration
