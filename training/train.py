from ultralytics import YOLO
import os
import torch
import random
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # YOLOv8 handles some determinism via `deterministic=True` arg in train()

def get_config():
    """
    Returns training configuration dictionary.
    """
    return {
        'model': 'yolov8n.pt',
        'data': os.path.abspath("training/dataset.yaml"),
        'epochs': 20, # Kept at 20 for verification speed, normally 50-100
        'imgsz': 640,
        'batch': 16, # Fixed batch for stability
        'device': 'cpu', # or 'mps', 'cuda'
        'project': 'models',
        'name': 'yolov8n_custom',
        'optimizer': 'auto',
        'seed': 42,
        'deterministic': True,
        # Augmentations
        'mosaic': 1.0,
        'mixup': 0.8,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 2.0,
        'perspective': 0.0001,
        'flipud': 0.0,
        'fliplr': 0.5,
        # Optimization
        'cos_lr': True,
        'amp': True,
        'exist_ok': True,
        'plots': True,
        'save': True,
    }

def train_model():
    # 1. Setup
    seed_everything(42)
    cfg = get_config()
    
    print(f"Starting training with config: {cfg}")
    
    # 2. Model
    model = YOLO(cfg['model'])

    # 3. Train
    # Passing cfg dict as kwargs
    results = model.train(**cfg)

    # 4. Export Best Model (Standardization)
    import shutil
    final_model_path = os.path.join(cfg['project'], cfg['name'], "weights", "best.pt")
    target_path = os.path.join(cfg['project'], "latest.pt")
    
    if os.path.exists(final_model_path):
        shutil.copy(final_model_path, target_path)
        print(f"Model saved to {target_path}")
    else:
        print("Warning: Best model weights not found.")

if __name__ == "__main__":
    train_model()
