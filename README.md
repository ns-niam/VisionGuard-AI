# VisionGuard AI

### Deep Learning-Based Supermarket CCTV Object Detection and Intelligent Video Analytics

<p align="center">
  <strong>Computer Vision | Deep Learning | Object Detection | Multi-Object Tracking</strong>
</p>

---

## 1. Overview

VisionGuard AI is a deep learning-based computer vision project designed to detect and analyze people in supermarket and retail surveillance footage.

The system aims to combine a fine-tuned YOLO object detection model with multi-object tracking and spatial analytics to provide meaningful insights from CCTV video streams.

The project focuses on building a reproducible object detection pipeline that can be trained, evaluated, and executed using Google Colab.

In addition to the core object detection task, the project explores advanced video analytics, including people counting, queue estimation, and zone monitoring.

## 2. Problem Statement

Supermarkets and retail environments use surveillance cameras to monitor customer activity and store operations.

However, manually reviewing surveillance footage is time-consuming and does not provide automated, real-time information about customer movement, occupancy, or queue conditions.

VisionGuard AI explores how deep learning and computer vision can automate these tasks through object detection and video analysis.

## 3. Project Objectives

The primary objectives are:

1. Develop a deep learning-based object detection model for retail surveillance footage.
2. Fine-tune a pretrained detector using a publicly available annotated dataset.
3. Evaluate model performance using standard object detection metrics.
4. Conduct a controlled experiment to investigate the effect of a selected training modification.
5. Build a reproducible training and evaluation pipeline.
6. Extend the trained detector with multi-object tracking and CCTV analytics.

## 4. System Scope

### 4.1 Core Requirements

The primary academic deliverable is a reproducible object detection system.

| Component | Description |
|---|---|
| Dataset | Public object detection dataset |
| Initial model | Pretrained YOLO11n candidate |
| Training | Fine-tuning using PyTorch |
| Baseline | Initial training configuration |
| Experiment | One controlled training modification |
| Evaluation | Precision, Recall, mAP@0.50 |
| Visualization | Ground-truth and predicted bounding boxes |
| Reproducibility | Google Colab training notebook |

### 4.2 Advanced Features

The following features are planned as extensions of the trained detection model.

| Feature | Purpose | Proposed Technology |
|---|---|---|
| Multi-object tracking | Track objects across video frames | ByteTrack |
| People counting | Estimate entry and exit counts | Tracking + line crossing |
| Queue estimation | Estimate customer queue occupancy | Tracking + ROI analysis |
| Zone monitoring | Monitor occupancy in selected areas | Polygon-based ROI |
| Movement visualization | Visualize tracked trajectories | OpenCV |

These features extend the core object detection pipeline and do not replace the required training and evaluation experiments.

## 5. Proposed System Architecture

```text
          CCTV Images / Recorded Videos
                       |
                       v
              Data Preprocessing
                       |
                       v
             YOLO Object Detector
                 (YOLO11n)
                       |
                       v
              Bounding Box Output
                       |
                       v
                ByteTrack
                       |
          +------------+------------+
          |            |            |
          v            v            v
       People       Queue        Zone
       Counting   Estimation   Monitoring
          |            |            |
          +------------+------------+
                       |
                       v
                Analytics Output
                       |
                       v
             Annotated Video / Reports
```

The initial detection model will focus on the person class.

Additional detection classes may be introduced if suitable datasets and compatible annotations are verified.

## 6. Dataset Strategy

The project will use publicly available object detection datasets with bounding-box annotations.

Dataset selection will consider:

- Relevance to retail and surveillance environments.
- Image quality and camera perspectives.
- Annotation accuracy.
- Dataset size and class distribution.
- Usage terms and licensing.
- Compatibility with the YOLO annotation format.

### 6.1 Dataset Candidates

| Dataset | Intended Purpose | Status |
|---|---|---|
| MALL-VLM | Surveillance-based person detection and tracking research | Under evaluation |
| RetailAction | Retail environment and human-object interaction research | Under evaluation |
| Shopping Trolley Detection | Optional trolley detection extension | Under evaluation |

Dataset sources:

- [MALL-VLM](https://github.com/harshm2601/mall-vlm-devkit)
- [RetailAction](https://huggingface.co/datasets/standard-cognition/RetailAction)
- [Shopping Trolley Detection](https://data.mendeley.com/datasets/vj9g4mn4vv/1)

No dataset will be used for training until its licensing, annotation format, and suitability have been checked.

### 6.2 Dataset Preparation

The preparation pipeline will include:

1. Dataset acquisition.
2. Annotation inspection.
3. Bounding-box validation.
4. Image and label consistency checks.
5. Duplicate and near-duplicate detection.
6. Train, validation, and test splitting.
7. Dataset statistics generation.
8. Ground-truth visualization.

The initial target is approximately 1,000 annotated images, subject to dataset availability and annotation quality.

The proposed split is:

| Split | Target Images |
|---|---:|
| Training | 700 |
| Validation | 150 |
| Testing | 150 |
| Total | 1,000 |

These are preliminary targets. Final counts will be determined after dataset auditing.

Data leakage will be avoided by keeping related frames, source videos, and identities within appropriate splits.

The final README will document actual image counts, object counts, class names, annotation format, split methodology, and random seed.

## 7. Deep Learning Model

### 7.1 Proposed Architecture

The initial model candidate is YOLO11n with pretrained weights.

Technology:

- PyTorch
- Ultralytics YOLO
- Transfer learning
- GPU-accelerated training

The initial model will focus on detecting people in surveillance images.

The final model configuration will be documented after dataset and environment compatibility checks.

### 7.2 Baseline Training

The baseline model will be fine-tuned on the training dataset.

The following parameters will be recorded:

| Parameter | Initial Proposal |
|---|---|
| Architecture | YOLO11n |
| Initial weights | Pretrained YOLO11n |
| Image size | 640 |
| Epochs | 50 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Random seed | 42 |
| Training device | Google Colab GPU |

These values are initial experimental settings and may be adjusted based on dataset size and available GPU memory.

Training evidence will include:

- Training configuration
- Training logs
- Learning curves
- Validation metrics
- Best model checkpoint
- Training duration

## 8. Controlled Experiment

A second training run will investigate the effect of one intentional modification.

### Proposed Experiment

**Research Hypothesis:**

Applying a small rotation augmentation may improve model generalization across different CCTV camera angles.

| Setting | Baseline | Experiment 02 |
|---|---|---|
| Model | YOLO11n | YOLO11n |
| Initial weights | Same | Same |
| Dataset split | Same | Same |
| Image size | 640 | 640 |
| Epochs | Same | Same |
| Batch size | Same | Same |
| Learning rate | Same | Same |
| Rotation augmentation | Default | 5 degrees |

Only the selected augmentation parameter will be changed.

Both experiments will be compared using validation precision, recall, mAP@0.50, and training time.

An improvement is not required. The objective is to conduct a valid controlled experiment and explain the observed results.

## 9. Model Evaluation

The final model will be selected using validation results.

The selected checkpoint will then be evaluated on the held-out test set.

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| Precision | Measures the proportion of predicted positives that are correct |
| Recall | Measures the proportion of actual objects detected |
| mAP@0.50 | Measures detection performance using an IoU threshold of 0.50 |

Evaluation settings will include the confidence threshold and IoU threshold.

### Required Visual Evidence

The final evaluation will include:

- At least five test images with predicted bounding boxes.
- Class names and confidence scores.
- Three representative failure cases.
- Explanation of likely failure causes.
- Practical suggestions for improvement.

Failure analysis will consider missed detections, false positives, incorrect classes, and inaccurate bounding boxes.

## 10. Advanced CCTV Analytics

After completing the required object detection experiments, the system will be extended with video analytics.

### 10.1 Multi-Object Tracking

ByteTrack will associate detections across consecutive video frames and assign tracking IDs.

### 10.2 People Counting

Virtual line-crossing logic will estimate directional entry and exit counts.

### 10.3 Queue Estimation

Checkout regions will be configured using polygon-based regions of interest.

The system will estimate the number of tracked people within selected queue regions.

### 10.4 Zone Monitoring

Configurable polygon regions will support occupancy monitoring in selected areas, such as:

- Store entrance
- Shopping area
- Checkout area
- Restricted area

These analytics will use detection and tracking outputs rather than requiring separate deep learning training.

## 11. Development Environment

Development and training will use separate environments.

| Environment | Responsibilities |
|---|---|
| GitHub Codespaces | Source code development, configuration, testing, and version control |
| Google Colab | Dataset preparation, GPU training, experiments, evaluation, and inference |

The training pipeline will be maintained in reusable Python modules.

The Colab notebook will execute the project pipeline from a fresh runtime using documented dependency versions.

## 12. Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Object Detection | Ultralytics YOLO |
| Model Candidate | YOLO11n |
| Object Tracking | ByteTrack |
| Video Processing | OpenCV |
| Numerical Computing | NumPy |
| Data Analysis | Pandas |
| Visualization | Matplotlib |
| Configuration | YAML |
| Development | GitHub Codespaces |
| GPU Training | Google Colab |
| Version Control | Git and GitHub |

## 13. Repository Structure

```text
VisionGuard-AI/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── dataset.yaml
│   ├── baseline.yaml
│   └── experiment.yaml
│
├── notebooks/
│   └── visionguard_training.ipynb
│
├── src/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── tracking/
│   ├── analytics/
│   └── visualization/
│
├── tests/
│
├── experiments/
│   ├── baseline/
│   └── experiment_02/
│
├── results/
│   ├── metrics/
│   ├── plots/
│   ├── predictions/
│   └── failure_analysis/
│
└── docs/
    └── methodology.md
```

## 14. Reproducibility

Reproducibility is a primary project objective.

The repository will provide:

- A documented dependency environment.
- Dataset acquisition instructions.
- Dataset and model configurations.
- A reproducible Google Colab notebook.
- Training and evaluation scripts.
- Random seed documentation.
- Experiment logs and learning curves.
- Accessible final model weights.
- Prediction examples and evaluation evidence.

Large datasets and model checkpoints may be hosted externally when necessary.

The README will include an Open in Colab link when the notebook is published.

## 15. Project Development Roadmap

| Phase | Work |
|---|---|
| 1 | Repository initialization |
| 2 | Dataset research and audit |
| 3 | Dataset preparation and validation |
| 4 | Baseline model training |
| 5 | Controlled experiment |
| 6 | Final test evaluation |
| 7 | Failure analysis and visual evidence |
| 8 | Advanced CCTV analytics |
| 9 | Documentation and reproducibility |
| 10 | Individual defense preparation |

## 16. Project Limitations

The initial model will detect only the classes represented in its training annotations.

Performance may vary depending on:

- Camera angle and resolution.
- Lighting conditions.
- Object occlusion.
- Detection distance.
- Dataset domain differences.
- Hardware and inference performance.

The initial implementation will use recorded video and image inputs.

Live CCTV or RTSP integration is an optional extension and is not required for the academic submission.

Tracking IDs are intended for temporary within-camera tracking, not personal identification.

## 17. Academic Information

**Course:** CSCI1409 - Deep Learning  
**Institution:** Kazakh-British Technical University  
**Project Type:** Individual Bonus Project  
**Submission Deadline:** 21 November 2026

The project follows the instructor's requirements for dataset preparation, baseline training, controlled experimentation, model evaluation, reproducibility, and individual defense.

AI-assisted development will be acknowledged in the final submission.

## 18. Project Status

**Current Stage:** Initial Repository Setup

Core model training, dataset verification, evaluation, and advanced analytics are planned development activities.

Experimental results and performance metrics will be published only after actual execution and verification.

---

## License

The license for original project code and any third-party dataset or model components will be documented separately.

Dataset and pretrained model usage must comply with their respective licenses.
