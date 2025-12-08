import cv2
import numpy as np

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (192, 192, 192), (128, 0, 0), (128, 128, 0), (0, 128, 0),
    (128, 0, 128), (0, 128, 128), (0, 0, 128)
]

def get_color(idx):
    return COLORS[idx % len(COLORS)]

class FusionVisualizer:
    def __init__(self):
        pass

    def draw_tracks(self, img, tracks):
        """
        Draw tracked objects on the frame.
        tracks: list of STrack objects (custom ByteTrack objects)
        """
        vis_img = img.copy()
        
        for t in tracks:
            if not t.is_activated:
                continue
            
            tid = t.track_id
            tlwh = t.tlwh
            x1, y1, w, h = map(int, tlwh)
            x2, y2 = x1 + w, y1 + h
            
            color = get_color(tid)
            
            # Box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"ID: {tid}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = x1 + text_size[0] + 3, y1 + text_size[1] + 4
            cv2.rectangle(vis_img, (x1, y1), c2, color, -1)
            cv2.putText(vis_img, label, (x1, y1 + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
            
        return vis_img

    def draw_detections(self, img, detections):
        """
        Draw raw detections (for debugging/comparison).
        detections: array of [x1, y1, x2, y2, conf, cls]
        """
        vis_img = img.copy()
        for det in detections:
             x1, y1, x2, y2 = map(int, det[:4])
             cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 0), 1) # Black thin box for raw detection
        return vis_img
