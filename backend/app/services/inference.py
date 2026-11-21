import logging
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.device = "cpu"

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        import torch
        from ultralytics import YOLO

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(self.model_path))
        logger.info("Loaded detection model on %s", self.device)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(
        self, image_bytes: bytes, confidence: float, iou: float, image_size: int
    ) -> dict:
        if self.model is None:
            raise RuntimeError("Inference model is not loaded")
        image = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
        start = time.perf_counter()
        result = self.model.predict(
            source=image,
            conf=confidence,
            iou=iou,
            imgsz=image_size,
            device=self.device,
            verbose=False,
        )[0]
        latency = (time.perf_counter() - start) * 1000
        names = result.names
        boxes = []
        if result.boxes is not None:
            for coords, score, class_id in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
                strict=True,
            ):
                class_id = int(class_id)
                boxes.append(
                    {
                        "class_id": class_id,
                        "class_name": str(names[class_id]),
                        "confidence": float(score),
                        "x1": float(coords[0]),
                        "y1": float(coords[1]),
                        "x2": float(coords[2]),
                        "y2": float(coords[3]),
                    }
                )
        annotated = result.plot()
        success, encoded = cv2.imencode(".jpg", annotated)
        if not success:
            raise RuntimeError("Unable to encode annotated image")
        return {
            "detections": boxes,
            "annotated_image": encoded.tobytes(),
            "latency_ms": latency,
        }
