import cv2
import numpy as np
import onnxruntime as ort
import torch
import time
from .utils import letterbox, non_max_suppression, scale_boxes, draw_detections, select_device

class Detector:
    def __init__(self, model_path, backend='onnx', conf_thres=0.25, iou_thres=0.45, device=None):
        """
        Initializes the Detector with specified backend.

        Args:
            model_path (str): Path to model file (.onnx or .pt)
            backend (str): 'onnx' or 'pytorch'
            conf_thres (float): Confidence threshold for NMS
            iou_thres (float): IoU threshold for NMS
            device (str): Device to run inference on ('cpu', 'cuda', 'mps') - Auto-selected if None
        """
        self.device = select_device(device)
        self.backend = backend
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model_path = model_path
        self.profiling_data = {'preprocess': 0.0, 'inference': 0.0, 'postprocess': 0.0, 'count': 0}
        
        print(f"Loading model {model_path} with backend {backend} on {self.device}...")

        if backend == 'onnx':
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda':
                providers.insert(0, 'CUDAExecutionProvider')
            elif self.device == 'mps':
                # Attempt to use CoreMLExecutionProvider or just fallback to CPU for stability on Mac/ONNX
                # MPS provider for ONNX is still very experimental/limited support
                if 'CoreMLExecutionProvider' in ort.get_available_providers():
                    providers.insert(0, 'CoreMLExecutionProvider')
            
            # Optimization Options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
        
        elif backend == 'pytorch':
            from ultralytics import YOLO
            self.model_wrapper = YOLO(model_path)
            self.model = self.model_wrapper.model
            self.model.to(device)
            self.model.eval()
        
        
        # Load Names
        self.names = {}
        try:
             # Try to load from dataset.yaml in training/ dir
             import os
             import yaml
             project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
             yaml_path = os.path.join(project_root, 'training', 'dataset.yaml')
             if os.path.exists(yaml_path):
                 with open(yaml_path, 'r') as f:
                     data = yaml.safe_load(f)
                     self.names = data.get('names', {})
             else:
                 # Fallback to COCO default names if file not found
                 self.names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'} 
                 print("Warning: dataset.yaml not found, using partial COCO defaults.")
                 
             # Ensure keys are ints
             self.names = {int(k): v for k, v in self.names.items()}
             
        except Exception as e:
            print(f"Error loading class names: {e}")

        # Warmup
        self.warmup()

    def warmup(self):
        """Runs a dummy inference to initialize internal graphs."""
        print("Warming up...")
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy_input)
        # Reset profiling after warmup
        self.profiling_data = {'preprocess': 0.0, 'inference': 0.0, 'postprocess': 0.0, 'count': 0}
        print("Warmup done.")

    def preprocess(self, img0):
        """
        Prepares input image for inference.
        
        Args:
            img0 (np.ndarray): Original input image (BGR)

        Returns:
            tensor/array: Normalized input
            ratio (tuple): Scaling ratio
            dwdh (tuple): Padding steps
        """
        # Resize/Pad
        img = letterbox(img0, new_shape=640, stride=32, auto=(self.backend=='pytorch'))[0]
        
        # HWC to CHW, BGR to RGB
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        
        # Normalization (0-255 to 0.0-1.0)
        img = img.astype(np.float32) / 255.0
        
        # Batch dimension
        if self.backend == 'onnx':
            img = img[None] # (1, 3, 640, 640)
        else:
            img = torch.from_numpy(img).to(self.device).float()
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
                
        return img, (img0.shape[:2], img.shape[2:]), None # Returning simpler ratio structure if needed, but sticking to legacy utils sig

    def inference(self, img_tensor):
        """Runs model inference."""
        if self.backend == 'onnx':
            out = self.session.run(None, {self.input_name: img_tensor})[0]
            # ONNX output is usually numpy
            # Convert to torch for consistent postprocess (NMS uses torch)
            return torch.from_numpy(out).to(self.device)
        else:
            with torch.no_grad():
                out = self.model(img_tensor)
                # Ultralytics model returns list of preds or tensor depending on head
                # Usually out[0] is the prediction tensor?
                # Direct model call usually returns a tuple/list or tensor. 
                # YOLOv8n head returns [tensor] list during eval?
                if isinstance(out, (list, tuple)):
                    return out[0]
                return out

    def postprocess(self, prediction, img0_shape, ratio, dwdh):
        """
        Applies NMS and scales boxes to original image.
        """
        # NMS
        preds = non_max_suppression(prediction, self.conf_thres, self.iou_thres)
        
        results = []
        for i, det in enumerate(preds):  # per image
            if len(det):
                # Scale boxes
                # Note: `utils.letterbox` returns (im, ratio, (dw, dh))
                # My `preprocess` didn't fully capture that return signature perfectly.
                # Assuming `ratio` passed here is just needed for `scale_boxes` which calculates gain internally if ratio_pad is None
                # Or we explicitly pass it. 
                # Let's rely on scale_boxes calculating from shapes.
                
                # Rescale boxes from img_size to im0 size
                # prediction is normalized or pixels? Pixels in 640x640.
                det[:, :4] = scale_boxes((640, 640), 
                                         det[:, :4], 
                                         img0_shape)
                results.append(det.cpu().numpy())
            else:
                results.append(np.empty((0, 6)))
        
        return results

    def detect(self, img0):
        """
        End-to-end detection pipeline.
        """
        # 1. Preprocess
        t1 = time.perf_counter()
        img_tensor, ratio, dwdh = self.preprocess(img0) # Re-using modified preprocess logic
        t2 = time.perf_counter()
        
        # 2. Inference
        raw_output = self.inference(img_tensor)
        t3 = time.perf_counter()
        
        # 3. Postprocess
        detections = self.postprocess(raw_output, img0.shape, ratio, dwdh)
        t4 = time.perf_counter()
        
        # Profile Update
        self.profiling_data['preprocess'] += (t2 - t1)
        self.profiling_data['inference'] += (t3 - t2)
        self.profiling_data['postprocess'] += (t4 - t3)
        self.profiling_data['count'] += 1
        
        return detections[0] # Return result for first image

    def get_avg_latency(self):
        """Returns dictionary of average latencies in ms."""
        if self.profiling_data['count'] == 0:
            return {}
        count = self.profiling_data['count']
        return {
            'preprocess': (self.profiling_data['preprocess'] / count) * 1000,
            'inference': (self.profiling_data['inference'] / count) * 1000,
            'postprocess': (self.profiling_data['postprocess'] / count) * 1000,
            'total': ((self.profiling_data['preprocess'] + self.profiling_data['inference'] + self.profiling_data['postprocess']) / count) * 1000
        }

def load_classes(yaml_path):
    import yaml
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

if __name__ == "__main__":
    # Test Block
    pass
