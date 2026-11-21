# VisionGuard-AI
### Deep Learning-Based Pedestrian Detection Using YOLO11 and the MOT17 Dataset

<p align="center">

**An Object Detection Research Project**

Built with PyTorch, Ultralytics YOLO11, and Google Colab

</p>

---

## 1. Project Overview

**VisionGuard-AI** is a deep learning-based pedestrian detection project developed as an individual bonus project for the CSCI1409 Deep Learning course at Kazakh-British Technical University (KBTU).

The project focuses on detecting pedestrians in surveillance-style images using transfer learning and pretrained YOLO11 object detection models.

The main objective is to develop, train, evaluate, and compare multiple object detection configurations while following a reproducible experimental workflow.

The project covers the complete object detection pipeline:

- Dataset acquisition and annotation processing.
- Dataset validation and sequence-wise splitting.
- Pretrained model fine-tuning.
- Controlled experimentation and hyperparameter optimization.
- Model selection using validation results.
- Independent evaluation on a held-out test set.
- Prediction visualization and failure analysis.
- Deployment-oriented inference preparation.

**Final selected model:** YOLO11s with a tuned learning rate of 0.0005.

## 2. Project Objectives

The primary objectives are:

1. Understand and prepare a real-world pedestrian detection dataset.
2. Fine-tune pretrained YOLO11 models using transfer learning.
3. Investigate the effect of image size and learning-rate changes.
4. Compare different model configurations using validation metrics.
5. Select the final model without using test results for model selection.
6. Evaluate the selected model on an independent test set.
7. Analyze prediction errors and identify model limitations.
8. Prepare reproducible training and inference artifacts.

## 3. Dataset

### 3.1 Dataset Source

The project uses **MOT17 (Multiple Object Tracking Benchmark 2017)**.

- Official dataset: https://motchallenge.net/data/MOT17/
- Hugging Face dataset mirror: https://huggingface.co/datasets/Lekim89/MOT17

MOT17 contains surveillance-style pedestrian sequences with bounding-box annotations.

Only the pedestrian class was used for this object detection task.

**Dataset usage:** The original MOTChallenge terms apply. Users must review and comply with the official dataset terms. The dataset is not redistributed through this repository.

### 3.2 Dataset Preparation

A documented subset of 2,000 images was selected from seven annotated MOT17 sequences.

| Split | Source sequences | Images | Annotated objects |
|---|---|---:|---:|
| Training | MOT17-02, 04, 05, 09 | 1,200 | 28,390 |
| Validation | MOT17-10 | 400 | 7,854 |
| Testing | MOT17-11, 13 | 400 | 5,216 |
| **Total** | **7 sequences** | **2,000** | **41,460** |

### 3.3 Split Strategy

The dataset was divided by source sequence rather than randomly splitting individual frames.

This approach prevents frames from the same source video from appearing across different subsets.

A deterministic temporal sampling strategy using `numpy.linspace` was applied to select frames from each sequence.

Training randomness was configured with seed `42`. Temporal frame selection itself did not use random sampling.

### 3.4 Annotation Format

Original MOT17 pedestrian annotations were converted into YOLO format.

Each label contains:

`class_id x_center y_center width height`

Bounding-box coordinates are normalized relative to image dimensions.

The pedestrian class mapping is:

| Class ID | Class |
|---:|---|
| 0 | person |

Bounding boxes were clipped to image boundaries, and fully invalid boxes were discarded.

### 3.5 Dataset Validation

The prepared dataset was audited for:

- Missing image-label pairs.
- Corrupted images.
- Invalid bounding boxes.
- Incorrect annotation structure.
- Split consistency.

Final validation:

| Check | Result |
|---|---:|
| Training images | 1,200 |
| Validation images | 400 |
| Test images | 400 |
| Missing labels | 0 |
| Corrupted images | 0 |
| Invalid annotations | 0 |
| Total valid annotations | 41,460 |

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning framework | PyTorch |
| Object detection framework | Ultralytics |
| Detection architecture | YOLO11 |
| Dataset | MOT17 |
| Annotation format | YOLO |
| Data processing | NumPy, OpenCV |
| Data analysis | Pandas |
| Visualization | Matplotlib |
| Training environment | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Development environment | GitHub Codespaces |
| Version control | Git and GitHub |

### 4.1 Training Environment

The main training environment was Google Colab.

| Component | Version / Specification |
|---|---|
| Python | 3.13.16 |
| PyTorch | 2.14.1+cu130 |
| Ultralytics | 8.4.215 |
| OpenCV | 4.14.0 |
| NumPy | 2.4.6 |
| Pandas | 2.2.3 |
| GPU | NVIDIA Tesla T4 |
| GPU Memory | Approximately 14.56 GB |

## 5. Model Architecture

The project uses pretrained YOLO11 object detection models.

YOLO11 is a one-stage object detector that predicts object locations and class probabilities in a single forward pass.

The project investigated the following configurations:

| Experiment | Model | Configuration |
|---|---|---|
| Baseline | YOLO11n | Pretrained, 640px |
| Experiment 02 | YOLO11n | Image size reduced to 512px |
| Refined | YOLO11n | Fine-tuning from baseline checkpoint |
| Model comparison | YOLO11s | Pretrained, 640px |
| Experiment 03 | YOLO11s | Learning rate tuned to 0.0005 |

The final selected model is **YOLO11s LR Tuned**.

## 6. Training Configuration

### 6.1 Baseline Configuration

| Parameter | Value |
|---|---|
| Model | YOLO11n |
| Initial weights | Pretrained YOLO11n |
| Image size | 640 × 640 |
| Batch size | 8 |
| Epochs | 15 |
| Optimizer | AdamW (resolved by auto configuration) |
| Learning rate | 0.002 |
| Random seed | 42 |
| Device | NVIDIA Tesla T4 |
| Training time | Approximately 8.75 minutes |

### 6.2 Controlled Experiment — Image Size

The controlled experiment investigated the impact of reducing image resolution.

**Hypothesis:** Reducing image size from 640 to 512 pixels may reduce computational cost and training time, but could affect detection performance, particularly for small or distant pedestrians.

Only the image size was intentionally changed.

| Parameter | Baseline | Experiment 02 |
|---|---:|---:|
| Model | YOLO11n | YOLO11n |
| Image size | 640 | 512 |
| Batch size | 8 | 8 |
| Epochs | 15 | 15 |
| Learning rate | 0.002 | 0.002 |
| Random seed | 42 | 42 |
| Training time | 8.75 min | 7.22 min |

### 6.3 Additional Experiments

Additional experiments investigated model capacity and learning-rate optimization.

**YOLO11n Refined**
- Initialized from the baseline checkpoint.
- Maximum additional epochs: 25.
- Learning rate: 0.0005.
- Early stopping was triggered after 13 additional epochs.

**YOLO11s**
- Pretrained YOLO11s initialization.
- Image size: 640.
- Batch size: 8.
- Maximum epochs: 40.
- Early stopping was triggered after 12 epochs.

**YOLO11s LR Tuned**
- Same YOLO11s architecture and pretrained initialization.
- Learning rate changed from 0.002 to 0.0005.
- Image size: 640.
- Batch size: 8.
- Maximum epochs: 40.
- Early stopping was triggered after 18 epochs.

## 7. Validation Results and Model Selection

All candidate checkpoints were evaluated using the same validation configuration:

- Image size: 640.
- Confidence threshold: 0.001.
- NMS IoU threshold: 0.70.

| Model | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|
| YOLO11s LR Tuned | 0.8394 | 0.4910 | 0.6196 | 0.5931 | 0.3017 |
| YOLO11s | 0.7541 | 0.4575 | 0.5695 | 0.5504 | 0.2685 |
| YOLO11n 512 Experiment | 0.7478 | 0.4427 | 0.5562 | 0.5202 | 0.2518 |
| YOLO11n Refined | 0.8001 | 0.4210 | 0.5517 | 0.4848 | 0.2383 |
| YOLO11n Baseline | 0.7533 | 0.4146 | 0.5348 | 0.4786 | 0.2296 |

YOLO11s LR Tuned achieved the strongest validation performance among the evaluated configurations.

Therefore, it was selected as the final model using validation results only.

## 8. Final Test Evaluation

The selected checkpoint was evaluated on the independent held-out test set.

**Final model:** YOLO11s LR Tuned

| Evaluation setting | Value |
|---|---|
| Test images | 400 |
| Ground-truth objects | 5,216 |
| Image size | 640 |
| Confidence threshold | 0.001 |
| NMS IoU threshold | 0.70 |

### Final Test Metrics

| Metric | Result |
|---|---:|
| Precision | **80.03%** |
| Recall | **43.19%** |
| F1-Score | **56.10%** |
| mAP@50 | **50.39%** |
| mAP@50-95 | **31.03%** |

### Metric Interpretation

- **Precision:** Measures how many predicted detections are correct.
- **Recall:** Measures how many actual objects are successfully detected.
- **F1-Score:** Harmonic mean of precision and recall.
- **mAP@50:** Mean Average Precision at an IoU threshold of 0.50.
- **mAP@50-95:** Average precision across IoU thresholds from 0.50 to 0.95.

The model achieved relatively high precision but lower recall, indicating that some pedestrians were missed.

The test evaluation above uses confidence threshold 0.001. A separate validation threshold analysis selected 0.05 based on validation F1. These configurations are reported separately to preserve the original held-out test evaluation.

## 9. Prediction Visualization

The final model was used to generate predictions on five representative test images.

Prediction visualizations include:

- Bounding boxes.
- Predicted class names.
- Detection confidence scores.

The project also includes a sample-level ground-truth versus prediction comparison.

The five-image comparison is qualitative evidence and is not a replacement for the full test-set metrics.

## 10. Failure Analysis

Three major categories of detection failures were identified.

### 10.1 Occlusion and Crowded Scenes

**Observation:** Pedestrians partially hidden behind other people or objects may not be detected.

**Possible cause:** Limited visible features and overlapping bounding boxes.

**Potential improvement:** Add more crowded-scene examples and evaluate suitable augmentation strategies.

### 10.2 Small and Distant Pedestrians

**Observation:** Small pedestrians in distant regions of an image are difficult to detect.

**Possible cause:** Limited spatial detail after image resizing.

**Potential improvement:** Investigate higher-resolution inference and multi-scale training.

### 10.3 False-Positive Detections

**Observation:** Some background regions are incorrectly detected as pedestrians.

**Possible cause:** Visual similarities between background structures and pedestrian features.

**Potential improvement:** Add representative hard-negative examples and evaluate confidence calibration.

These explanations are plausible interpretations of observed failures, not experimentally proven causes.

## 11. Inference and Deployment Preparation

A production-oriented inference prototype was prepared using the final trained checkpoint.

The inference pipeline supports:

- Image validation.
- Single-image inference.
- Batch inference.
- Configurable confidence threshold.
- Configurable NMS IoU threshold.
- Structured detection outputs.
- Bounding-box visualization.
- Inference timing.

### Benchmark Results

The final model was benchmarked on 32 test images using an NVIDIA Tesla T4.

| Batch Size | Throughput |
|---:|---:|
| 1 | 10.66 images/sec |
| 4 | 40.55 images/sec |
| 8 | 38.14 images/sec |
| 16 | 43.13 images/sec |

The highest measured throughput was achieved with batch size 16.

These are experimental measurements from one Colab T4 runtime, not production service-level guarantees.

A model card, inference configuration, integrity audit, and deployment archive were also prepared.

**Note:** The project currently provides a deployment-oriented inference prototype and model package. A fully hosted production service with API authentication, monitoring, and infrastructure has not been implemented.

## 12. Repository Structure

The repository is intended to follow this structure:

```text
VisionGuard-AI/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── VisionGuard_AI.ipynb
│
├── configs/
│   ├── dataset.yaml
│   └── inference_config.json
│
├── models/
│   └── best.pt
│
├── src/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   └── inference/
│
├── reports/
│   ├── baseline_training_report.json
│   ├── final_test_evaluation.json
│   └── failure_analysis.md
│
├── metrics/
│
├── predictions/
│
├── visualizations/
│
└── deployment/
    └── MODEL_CARD.md
```

Large dataset files and model weights may be hosted externally rather than committed directly to GitHub.

## 13. Installation

Clone the repository:

```bash
git clone https://github.com/ns-niam/VisionGuard-AI.git
cd VisionGuard-AI
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The exact dependency requirements will be documented in the repository.

## 14. Running Inference

After obtaining the final checkpoint, place it at:

```text
models/best.pt
```

The inference configuration and example usage will be documented alongside the inference source code.

Training and GPU-intensive experiments were performed in Google Colab. GitHub Codespaces is used for source-code development, repository management, and lightweight testing.

## 15. Reproducibility and Artifacts

The project preserves the following experimental artifacts:

- Dataset split manifest.
- Dataset annotation statistics.
- Model training configurations.
- Baseline and experiment training reports.
- Training logs and learning curves.
- Validation comparison metrics.
- Final test evaluation report.
- Prediction visualizations.
- Failure analysis.
- Model integrity audit.
- Performance benchmark.
- Model card and deployment package.

The final notebook is intended to provide the complete training and evaluation workflow from a fresh Google Colab runtime.

### Important Links

| Resource | Link |
|---|---|
| GitHub Repository | https://github.com/ns-niam/VisionGuard-AI |
| Google Colab Notebook | Add notebook link here |
| Final Model Checkpoint | Add accessible checkpoint link here |
| Dataset Source | https://motchallenge.net/data/MOT17/ |

## 16. Limitations and Future Work

Current limitations include:

- Lower recall in crowded and occluded scenes.
- Difficulty detecting small and distant pedestrians.
- False-positive detections in complex backgrounds.
- Evaluation limited to a selected subset of MOT17.
- Performance measurements obtained from a single Colab GPU environment.

Future improvements may include:

1. Higher-resolution training and inference.
2. More advanced data augmentation.
3. Additional hard-negative training examples.
4. Evaluation of alternative detection architectures.
5. Multi-object tracking integration.
6. Video-based pedestrian counting and occupancy analytics.
7. FastAPI-based inference service.
8. Docker-based deployment and monitoring.

## 17. Academic Information

| Field | Information |
|---|---|
| Project | VisionGuard-AI |
| Course | CSCI1409 — Deep Learning |
| Institution | Kazakh-British Technical University |
| Project Type | Individual Bonus Project |
| Student | MD Sha Niamatullah |
| Academic Term | Fall 2026 |
| Submission Date | 21 November 2026 |

### Acknowledgment of AI Assistance

AI-assisted tools were used to support code development, debugging, documentation, and technical explanations. Dataset preparation, model training, experiments, evaluation, and reported results were carried out as part of this project.

---

## License and Dataset Terms

The original project code and third-party components must be considered separately.

MOT17 dataset usage is governed by the original MOTChallenge terms. Users must obtain the dataset from an authorized source and comply with its conditions.

Pretrained model and framework usage is subject to their respective licenses and terms.