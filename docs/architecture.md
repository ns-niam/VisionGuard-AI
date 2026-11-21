# VisionGuard-AI application architecture

The application keeps the research project as the source of truth and adds a thin production layer around the final checkpoint.

```text
Next.js dashboard -> FastAPI API -> inference service -> models/final/best.pt
                                  |-> SQLAlchemy repository -> PostgreSQL / SQLite
```

The model is loaded once in the FastAPI lifespan. Uploads are validated in memory, inference returns structured boxes and an annotated JPEG, and only the detection record and annotated image are persisted. Uploaded originals are never exposed as public files. The default database is SQLite for local development; Docker Compose uses PostgreSQL.

The selected model remains YOLO11s LR Tuned. The model path is configurable and the checkpoint is mounted into the container rather than copied into its image.
