# VisionGuard-AI Model Card

## Model Information

- Model: YOLO11s LR Tuned
- Task: Pedestrian Object Detection
- Framework: PyTorch and Ultralytics
- Architecture: YOLO11s
- Parameters: 9,428,179
- Checkpoint size: 18.263 MB
- Classes: Person

## Intended Use

Detect pedestrians in CCTV-style images and recorded surveillance footage.

This model is an academic research prototype. It is not validated for safety-critical or fully autonomous surveillance decisions.

## Dataset

- Dataset: MOT17
- Selected images: 2,000
- Training images: 1,200
- Validation images: 400
- Test images: 400
- Split strategy: Sequence-wise custom split
- Annotation format: YOLO

The custom split is not an official MOT17 benchmark evaluation.

Dataset usage must comply with the original MOTChallenge terms.

## Test Evaluation

- Precision: 0.8003348955845689
- Recall: 0.4318852085400249
- mAP@0.50: 0.5039471142285958
- mAP@0.50:0.95: None
- Confidence threshold: 0.001
- NMS IoU threshold: 0.70
- Image size: 640

## Inference Configuration

The recommended deployment inference settings are documented in `inference_config.json`.

The confidence threshold of 0.05 was selected using validation data.

## Performance Benchmark

Hardware: Tesla T4

Benchmark image count: 32

The benchmark measures inference throughput and batch latency under the recorded Colab runtime conditions. Results may vary across hardware and environments.

## Limitations

- Reduced recall in crowded scenes.
- Occlusion can cause missed detections.
- Small and distant pedestrians may be difficult to detect.
- Performance may vary with camera angle, lighting, and image quality.
- No identity recognition or long-term tracking is provided.
- No live CCTV integration is included.

## Ethical Considerations

The model detects the visual class person. It does not identify individuals.

Use should comply with applicable privacy laws and surveillance policies.

## Reproducibility

The package includes the trained checkpoint, inference configuration, evaluation reports, and available experiment metadata.

Generated: 2026-11-21T00:42:45.194946+00:00
