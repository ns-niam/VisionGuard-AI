# VisionGuard-AI — Failure Case Analysis

## Method
Five sampled test images were compared against their ground-truth annotations.
Predictions were matched one-to-one using IoU >= 0.50.

## Sample Summary

| Metric | Count |
|---|---:|
| Ground-truth objects | 73 |
| Predictions | 81 |
| Matched predictions | 33 |
| False positives | 48 |
| Missed objects | 40 |

## Failure Case 1: Crowded Scenes
Image: `MOT17-11-FRCNN_000900.jpg`

Likely issues include pedestrian overlap, occlusion and difficult localization.

Practical next step: Inspect crowded examples and investigate higher-resolution training.

## Failure Case 2: Small and Distant Pedestrians
Image: `MOT17-13-FRCNN_000373.jpg`

Small objects and dense pedestrian groups may contribute to missed detections.

Practical next step: Investigate multi-scale training and small-object detection.

## Failure Case 3: Background False Detections
Image: `MOT17-13-FRCNN_000750.jpg`

Some predictions do not align with ground-truth pedestrians and show inaccurate localization.

Practical next step: Inspect hard-negative background examples and localization errors.

## Limitations
This is a qualitative analysis of five sampled test images, not the complete test set.
No further model tuning is performed using the test set.
