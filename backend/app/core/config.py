import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings:
    app_name: str = os.getenv("APP_NAME", "VisionGuard-AI API")
    environment: str = os.getenv("ENVIRONMENT", "development")
    database_url: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'backend' / 'visionguard.db'}"
    )
    model_path: Path = Path(
        os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "final" / "best.pt"))
    )
    cors_origins: list[str] = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ]
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    min_image_dimension: int = int(os.getenv("MIN_IMAGE_DIMENSION", "32"))
    max_image_dimension: int = int(os.getenv("MAX_IMAGE_DIMENSION", "10000"))
    default_confidence: float = float(os.getenv("DEFAULT_CONFIDENCE", "0.05"))
    default_iou: float = float(os.getenv("DEFAULT_IOU", "0.7"))
    default_image_size: int = int(os.getenv("DEFAULT_IMAGE_SIZE", "640"))


settings = Settings()
