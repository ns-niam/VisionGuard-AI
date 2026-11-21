from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DetectionBox(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)


class DetectionResponse(BaseModel):
    id: UUID
    filename: str
    content_type: str
    file_size: int
    image_width: int
    image_height: int
    detection_count: int
    confidence_threshold: float
    iou_threshold: float
    image_size: int
    inference_latency_ms: float
    detections: list[DetectionBox]
    annotated_image: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class DetectionSummary(BaseModel):
    id: UUID
    filename: str
    image_width: int
    image_height: int
    detection_count: int
    inference_latency_ms: float
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class DetectionListResponse(BaseModel):
    items: list[DetectionSummary]
    total: int


class Thresholds(BaseModel):
    confidence: float = Field(default=0.05, ge=0.001, le=1)
    iou: float = Field(default=0.7, ge=0.05, le=0.95)
    image_size: int = Field(default=640, ge=32, le=2048)
