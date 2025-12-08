import pytest
from inference.traffic_counter import VehicleCounter

# Mock Track Object
class MockTrack:
    def __init__(self, track_id, cls_name):
        self.track_id = track_id
        self.cls_name = cls_name

def test_initial_state():
    counter = VehicleCounter()
    assert counter.get_total_count(0) == 0
    assert counter.get_density_level(0) == "Low"

def test_counting_unique():
    counter = VehicleCounter(window_seconds=10)
    
    # t=1: Car 1 appears
    t1 = [MockTrack(1, "car")]
    counter.update(t1, 1.0)
    assert counter.get_total_count(1.0) == 1
    assert counter.get_window_counts(1.0) == {"car": 1}

    # t=2: Car 1 appears again (should not double count)
    counter.update(t1, 2.0)
    assert counter.get_total_count(2.0) == 1
    
    # t=3: Car 2 appears
    t2 = [MockTrack(2, "bus")]
    counter.update(t2, 3.0)
    assert counter.get_total_count(3.0) == 2
    assert counter.get_window_counts(3.0) == {"car": 1, "bus": 1}

def test_window_expiry():
    counter = VehicleCounter(window_seconds=5)
    
    # t=0: Car 1
    counter.update([MockTrack(1, "car")], 0.0)
    assert counter.get_total_count(0.0) == 1
    
    # t=4: Still valid
    assert counter.get_total_count(4.0) == 1
    
    # t=6: Expired (0.0 < 6.0 - 5.0)
    # Need to trigger prune via get or update
    assert counter.get_total_count(6.0) == 0

def test_density_levels():
    # Thresh: Low <= 2, Medium <= 4, High > 4 (Custom for test)
    counter = VehicleCounter(low_threshold=2, high_threshold=4)
    
    # 0 -> Low
    assert counter.get_density_level(0) == "Low"
    
    # 2 -> Low
    for i in range(2):
        counter.update([MockTrack(i, "car")], 0)
    assert counter.get_density_level(0) == "Low"
    
    # 3 -> Medium
    counter.update([MockTrack(2, "car")], 0)
    assert counter.get_density_level(0) == "Medium"
    
    # 5 -> High
    for i in range(3, 5):
        counter.update([MockTrack(i, "car")], 0)
    assert counter.get_density_level(0) == "High"

def test_ignore_untracked_classes():
    counter = VehicleCounter(tracked_classes=("car",))
    
    # Bus should be ignored
    counter.update([MockTrack(1, "bus")], 0)
    assert counter.get_total_count(0) == 0
    
    # Car should be counted
    counter.update([MockTrack(2, "car")], 0)
    assert counter.get_total_count(0) == 1
