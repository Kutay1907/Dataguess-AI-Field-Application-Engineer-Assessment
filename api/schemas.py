from typing import List, Optional
from pydantic import BaseModel, Field

class Detection(BaseModel):
    """
    Represents a single detected object.
    """
    box: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]", min_items=4, max_items=4)
    confidence: float = Field(..., description="Confidence score (0-1)")
    class_id: int = Field(..., description="Class ID")
    label: Optional[str] = Field(None, description="Class label (e.g. 'person')")

class DetectionResponse(BaseModel):
    """
    Response model for the /detect endpoint.
    """
    detections: List[Detection] = Field(default_factory=list, description="List of detections")
    count: int = Field(..., description="Total number of detections")
    inference_time: float = Field(..., description="Inference time in seconds")
    image_size: List[int] = Field(..., description="Image size [width, height]")
    backend: str = Field(..., description="Backend used for inference (onnx/pytorch)")
    traffic_total_count: int = Field(0, description="Unique vehicles in window")
    traffic_density_level: str = Field("Low", description="Traffic density level")
    traffic_per_class: dict[str, int] = Field({}, description="Counts per class in window")
