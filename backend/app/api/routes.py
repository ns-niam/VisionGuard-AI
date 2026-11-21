import base64
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from ..core.config import settings
from ..core.database import get_db
from ..repositories import detections as repository
from ..schemas.detection import (
    DetectionListResponse,
    DetectionResponse,
    DetectionSummary,
    Thresholds,
)
from ..services.image_validation import read_validated_image
from ..services.inference import InferenceService

router = APIRouter(prefix="/api/v1")


def get_inference_service() -> InferenceService:
    from ..main import inference_service

    return inference_service


@router.get("/health")
def health():
    service = get_inference_service()
    return {"status": "ok", "model_loaded": service.is_loaded, "device": service.device}


@router.get("/model/info")
def model_info():
    return {
        "name": "YOLO11s LR Tuned",
        "architecture": "YOLO11s",
        "checkpoint": "models/final/best.pt",
        "parameters": 9428179,
        "classes": [{"id": 0, "name": "person"}],
        "inference_defaults": {
            "image_size": 640,
            "confidence": settings.default_confidence,
            "iou": settings.default_iou,
        },
        "validation": {
            "precision": 0.8394,
            "recall": 0.4910,
            "f1": 0.6196,
            "map50": 0.5931,
            "map50_95": 0.3017,
            "split": "validation",
        },
        "held_out_test": {
            "precision": 0.8003,
            "recall": 0.4319,
            "f1": 0.5610,
            "map50": 0.5039,
            "map50_95": 0.3103,
            "split": "held-out test",
        },
    }


@router.post(
    "/detection/image",
    response_model=DetectionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def detect_image(
    file: UploadFile = File(...),  # noqa: B008
    thresholds: Thresholds = Depends(),  # noqa: B008
    db: Session = Depends(get_db),  # noqa: B008
):
    data, width, height = await read_validated_image(file)
    try:
        result = get_inference_service().predict(
            data, thresholds.confidence, thresholds.iou, thresholds.image_size
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail="Inference service is unavailable"
        ) from exc
    record = repository.create(
        db,
        filename=file.filename or "upload",
        content_type=file.content_type or "application/octet-stream",
        file_size=len(data),
        image_width=width,
        image_height=height,
        detection_count=len(result["detections"]),
        confidence_threshold=thresholds.confidence,
        iou_threshold=thresholds.iou,
        image_size=thresholds.image_size,
        inference_latency_ms=result["latency_ms"],
        detections=result["detections"],
        annotated_image=result["annotated_image"],
    )
    return {
        **record.__dict__,
        "annotated_image": "data:image/jpeg;base64,"
        + base64.b64encode(record.annotated_image).decode("ascii"),
    }


@router.get("/detections", response_model=DetectionListResponse)
def list_detections(
    search: str | None = Query(default=None, max_length=100),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),  # noqa: B008
):
    records, total = repository.list_records(db, search, offset, limit)
    return {
        "items": [DetectionSummary.model_validate(record) for record in records],
        "total": total,
    }


@router.get("/detections/{detection_id}", response_model=DetectionResponse)
def get_detection(detection_id: UUID, db: Session = Depends(get_db)):  # noqa: B008
    record = repository.get(db, detection_id)
    if not record:
        raise HTTPException(status_code=404, detail="Detection not found")
    return {
        **record.__dict__,
        "annotated_image": "data:image/jpeg;base64,"
        + base64.b64encode(record.annotated_image).decode("ascii"),
    }


@router.delete("/detections/{detection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_detection(detection_id: UUID, db: Session = Depends(get_db)):  # noqa: B008
    record = repository.get(db, detection_id)
    if not record:
        raise HTTPException(status_code=404, detail="Detection not found")
    repository.delete(db, record)
