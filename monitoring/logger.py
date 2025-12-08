import logging
import json
import sys
import os
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """
    Custom formatter to output JSON logs.
    """
    def format(self, record):
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        
        # Merge extra attributes if valid JSON compatible types
        if hasattr(record, "data") and isinstance(record.data, dict):
            log_obj.update(record.data)
            
        return json.dumps(log_obj)

def setup_logger(name="edge_ai", log_file="logs/system.log", level=logging.INFO):
    """
    Configures a logger with JSON formatting.
    """
    # Create logs directory if needed
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
        
    # JSON Formatter
    formatter = JSONFormatter()
    
    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console Handler (Optional: Keep it simple or also JSON)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
