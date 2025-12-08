import collections
from typing import List, Dict, Tuple, Any

class VehicleCounter:
    def __init__(
        self,
        window_seconds: float = 60.0,
        fps_hint: float = 25.0,
        tracked_classes: tuple = ("car", "truck", "bus", "motorcycle"),
        low_threshold: int = 20,
        high_threshold: int = 50,
    ):
        """
        Traffic Analytics Counter.
        Maintains a rolling window of unique vehicle counts and computes density.
        """
        self.window_seconds = window_seconds
        self.tracked_classes = set(tracked_classes)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        # Store events as (timestamp, track_id, cls_name)
        # Using deque for efficient popping from the left (oldest events)
        self.events = collections.deque()
        # Cache for O(1) lookup of currently active IDs in window
        self.active_tracks_cache = {}
        
    def update(self, tracks: List[Any], timestamp: float) -> None:
        """
        Consume current frame tracks.
        Only consider classes in self.tracked_classes.
        Append events for new track ids at this timestamp.
        
        Args:
            tracks: List of track objects or dicts. Must have 'track_id' and 'cls_name' (or 'label').
                    If objects, assuming attributes. If dicts, keys.
                    Ideally standardized. VideoEngine passes lists of STrack? 
                    Let's support objects with attributes for now per typical ByteTrack usage.
            timestamp: Current time in seconds.
        """
        # Prune old events -> efficient window maintenance
        self._prune(timestamp)
        
        # Get set of currently known unique (track_id, cls_name) into the window
        # But wait, we need to know if we *already* counted this track_id in the *current window*.
        # Actually, the requirement says "No double counting of the same vehicle in the window".
        # This means unique set of IDs present in the window.
        
        # Strategy:
        # We store *every* observation? No, that's too much data if 25 FPS * 60s.
        # We only need to store the *first time* we saw a track_id in the current window? 
        # But if the window slides, the "first time" might fall out.
        
        # Correction: The prompt suggests:
        # "Maintain an internal list... At each update call, push new events for tracks that are newly observed in the window."
        # Use a temporary set to check uniqueness against what's already in the window?
        
        # Better approach for efficient sliding window unique count:
        # 1. Store (timestamp, track_id, class_name) for EVERY frame? No.
        # 2. Store (timestamp, track_id, class_name) only when it's "freshly" seen?
        #    If I see ID 100 at t=0, I store it.
        #    At t=0.1, I see ID 100 again. If I don't store it, and t=0 falls out of window at t=60, 
        #    I lose ID 100 even if it was still visible at t=59. 
        
        # Re-reading prompt logic: "Regularly drop events that are older than now - window_seconds."
        # constraint: "For each (track_id, class_name) pair, only count it once per window."
        # This implies the `get_window_counts` method does the deduplication on fly from the event history.
        # So we SHOULD store observations. But storing every frame is heavy.
        
        # Optimizaton:
        # Update the 'last_seen' timestamp of a track?
        # If we just keep a dict of {track_id: last_seen_timestamp}, we can count how many have last_seen > now - window.
        # But we also need to know the class.
        
        # Let's effectively use a Dict[track_id, (timestamp, cls_name)].
        # When we see a track, we update its timestamp to `now`.
        # When we want to count, we filter this dict for timestamp > now - window.
        # Pruning can happen periodically on this dict.
        
        # Let's follow this "Registry" approach as it's cleaner than a raw list of events.
        
        for t in tracks:
            # Handle object vs dict
            if isinstance(t, dict):
                tid = t.get('track_id')
                cls = t.get('cls_name') or t.get('label')
            else:
                tid = getattr(t, 'track_id', None)
                # Fallback for class name mapping if needed, but assuming engine handles it
                # For now, let's try 'cls_name', 'label', or 'class_id'
                cls = getattr(t, 'cls_name', getattr(t, 'label', None))
                if cls is None and hasattr(t, 'class_id'):
                     # If raw class_id, we might need mapping, but let's assume string for now as per requirement 1.
                     cls = str(t.class_id)

            if tid is None or cls is None:
                continue
                
            if str(cls) not in self.tracked_classes:
                continue
                
            # Update registry
            # We treat track_id as unique key.
            # Using append to deque for the prompt's "list of events" style if requested strictly?
            # Prompt says: "Maintain an internal list... push new events... drop events older".
            # It also says: "For each pair, only count it once per window... maintain a temporary set... when computing".
            
            # The prompt strongly suggests a log-based approach. But the log-based approach has the "flickering" issue 
            # (if I only log start, and start falls out, count drops even if car is there).
            # UNLESS "newly observed" means "observed in this frame". -> Then we log every frame.
            # "Push new events for tracks that are newly observed IN THE WINDOW".
            # This is ambiguous. If I see car A at t=0, and it stays till t=10.
            # If I only log at t=0. 
            # At t=61 (window 60s), the t=0 event dies. Car A is not counted.
            # But maybe car A left at t=10. So it SHOULD not be counted at t=61 (Last seen 51s ago).
            # Wait, "Traffic Density" usually means "How many cars are here *right now* + recent history?" 
            # OR "Volume (Flow) per minute"?
            
            # "Counts vehicles per unit time" -> usually Flow (Flux).
            # "Density level" -> usually Concentration (Objects / Distance).
            
            # Re-reading: "Count unique vehicles in a sliding time window".
            # If a car passes in 5 sec, it counts as 1 for the next 60 seconds? Yes.
            # This measures "Flow volume per minute". This is standard for "Vehicles last 60 seconds".
            
            # So: We only record the FIRST time we see a track, or update it?
            # If we measure FLOW: We count "Unique IDs that appeared in [limit, now]".
            # So if ID 5 appeared at t=10, it contributes to count until t=70.
            # Even if it leaves screen at t=12.
            # This is consistent with "Vehicles per minute".
            
            # So we strictly implement:
            # - Store "First Timestamp" of a track_id. 
            # - Or simpler: Just store a set of active IDs in window?
            # - No, we need to expire them.
            
            # Let's stick to the prompt's likely intent:
            # Keep a history of (track_id, timestamp_seen).
            # BUT efficient way:
            # `self.vehicle_registry = {track_id: first_seen_timestamp}`?
            # If we want "Flow per min": Yes.
            # If we want "Current Occupancy": We need last_seen.
            
            # Prompt: "Counts vehicles per unit time ... rolling time window"
            # This confirms FLOW.
            
            # Implementation:
            # Use a dictionary `self.seen_tracks = {track_id: timestamp_last_seen}`?
            # No.
            # Let's use `self.history` deque of `(timestamp, track_id, class_name)`.
            # We record a track ONLY IF we haven't seen it "recently" (i.e. currently in window)?
            # But checking the whole deque is O(N).
            
            # OPTIMIZED HYBRID APPROACH:
            # `self.active_tracks_cache`: dict = {track_id: timestamp_added}
            # `self.events`: deque = [(timestamp, track_id, cls)] (for easy time-based purging)
            
            # When track T arrives:
            #   If T in `active_tracks_cache`:
            #       Ignore (already counted in this window)
            #   Else:
            #       Add to `active_tracks_cache`
            #       Add to `events`
            
            # On `_prune(now)`:
            #   While events[0].time < now - window:
            #       pop event
            #       remove event.track_id from `active_tracks_cache`
            
            # This is O(1) mostly.
            
            if tid in self.active_tracks_cache:
                continue
            
            # New unique vehicle for this window
            self.active_tracks_cache[tid] = timestamp
            self.events.append((timestamp, tid, str(cls)))

    def _prune(self, now: float):
        limit = now - self.window_seconds
        while self.events and self.events[0][0] < limit:
            ts, tid, cls = self.events.popleft()
            # Remove from cache to allow re-counting if it appears again later?
            # Usually unique IDs don't reappear after 60s (ByteTrack IDs typically increment or reset only on restart).
            # If ID reuse happens, we might double count. Assuming IDs are unique per session usually.
            if tid in self.active_tracks_cache:
                del self.active_tracks_cache[tid]

    def get_window_counts(self, now: float) -> Dict[str, int]:
        """
        Return per class counts for the current window.
        """
        self._prune(now) # Lazy prune ensure freshness
        
        counts = collections.defaultdict(int)
        for ts, tid, cls in self.events:
            counts[cls] += 1
        return dict(counts)

    def get_total_count(self, now: float) -> int:
        """
        Return total number of distinct vehicles in the current window.
        """
        self._prune(now)
        return len(self.active_tracks_cache)

    def get_density_level(self, now: float) -> str:
        """
        Compute traffic density level.
        Low: 0-20, Medium: 21-50, High: >50
        """
        total = self.get_total_count(now)
        if total <= self.low_threshold:
            return "Low"
        elif total <= self.high_threshold:
            return "Medium"
        else:
            return "High"

    # Initialize cache in __init__
    # (Doing it here to keep class consolidated in overwrite)
    # Re-writing __init__ part of the file correctly below.
