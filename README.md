# 🩺 Breast Cancer Detection and Localization Using Ultralytics YOLO

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8%2F26-green.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, production-ready computer vision pipeline designed for automated breast cancer detection and spatial localization using state-of-the-art **Ultralytics YOLO** architecture. This repository provides end-to-end code for configuration, training, quantitative evaluation on independent test splits, and model explainability via Grad-CAM heatmaps.

---

## 📖 Abstract

Breast cancer remains one of the most prevalent and life-threatening malignancies globally, where early detection significantly improves patient survival rates. Traditional manual screening methods are often time-consuming and prone to subjective diagnostic variability. This project presents an automated computer-aided detection (CAD) pipeline leveraging state-of-the-art **Ultralytics YOLO** architecture to accurately detect and localize cancerous lesions in medical scans. Utilizing a robustly split dataset comprising train, validation, and test subsets, the proposed model learns localized spatial patterns to differentiate between `cancer` and `normal` findings. Experimental evaluations demonstrate high detection accuracy, achieving an overall Box mAP50 of **94.76%** on independent test splits, illustrating the framework's potential as a reliable clinical decision-support tool.

## ⚙️ Method

## 🔎 Introduction

Computer Vision (CV) and deep learning have revolutionized medical image analysis by providing fast, objective, and consistent evaluations. Object detection models, particularly the YOLO (You Only Look Once) family, offer a powerful balance between high inference speed and precise spatial localization, making them ideal for clinical environments requiring real-time screening assistance. The primary objective of this repository is to establish an end-to-end open-source framework for breast cancer localization. By casting lesion detection as a bounding-box regression and classification problem, the system automates the identification of suspicious regions, minimizing human oversight and assisting radiologists in high-throughput diagnostic workflows.

## 📊 Dataset Overview

The dataset used in this project was collected from an online medical imaging repository and structured specifically for object detection tasks using the Ultralytics framework via a standard `data.yaml` configuration file.

* **Data Collection Source:** Collected from open-access online medical imaging resources.
* **Target Classes (`nc=2`):** `cancer`, `normal`[cite: 4].
* **Dataset Splits:**
  * **Train Set:** `985` images[cite: 4]
  * **Validation Set:** `164` images[cite: 4]
  * **Test Set:** `493` images[cite: 4]

The workflow is structured into four main phases:

1. **Dataset Preparation & Configuration:** The data is configured via a structured `data.yaml` schema containing two primary target classes (`cancer` and `normal`)[cite: 4]. The dataset is cleanly portioned into training (`985` images), validation (`164` images), and test (`493` images) splits to prevent data leakage and ensure robust generalization[cite: 4].
2. **Model Architecture & Training:** Training is initialized using pre-trained weights (`yolo26n.pt`) optimized via stochastic gradient descent configurations. Standard augmentation techniques—including mosaic framing and random scaling—are applied at an image resolution of `640x640` across `50` epochs[cite: 4].
3. **Independent Test Evaluation:** Post-training evaluation is executed strictly on the unseen test split using metrics such as Mean Average Precision (mAP), Precision, and Recall.
4. **Explainability & Heatmaps/Grad-CAM:** To bridge the gap between black-box deep learning and clinical trust, Ultralytics Solutions and Grad-CAM integration are implemented to visualize attention heatmaps that map out specific regions driving model predictions.

---

## 📊 Results & Performance Evaluation

The model was trained for **50 epochs** with an image size of `640x640`[cite: 4]. Final testing on an independent test dataset yielded exceptional detection performance:

| Metric                 | Score                         | Description                                                        |
| :--------------------- | :---------------------------- | :----------------------------------------------------------------- |
| **Box mAP50**    | **94.76%** (`0.9476`) | Mean Average Precision at IoU threshold 0.50[cite: 4]              |
| **Box mAP50-95** | **75.20%** (`0.7520`) | Mean Average Precision across IoU thresholds 0.50 to 0.95[cite: 4] |
| **Precision**    | **93.80%** (`0.9380`) | Accuracy of positive predictions[cite: 4]                          |
| **Recall**       | **91.53%** (`0.9153`) | Ability to detect all true instances[cite: 4]                      |

### Class-wise Performance Breakdown:

* **Cancer Class:** Box mAP50 = `84.1%` (Precision: `97.7%`, Recall: `80.2%`)[cite: 4]
* **Normal Class:** Box mAP50 = `90.3%` (Precision: `86.8%`, Recall: `95.6%`)[cite: 4]

---

## 🖼️ Visualizations & Model Predictions

### 1. Model Predictions (3x3 Grid)

Below is a 3x3 sample visualization grid showcasing bounding-box predictions generated by the YOLO model on test split images.

<p align="center">
  <img src="assets/prediction_3x3_grid.jpg" width="800" alt="3x3 Prediction Grid">
</p>

### 2. Explainability: Grad-CAM & Heatmaps

To interpret model decisions, attention heatmaps and Grad-CAM visualizations highlight the specific regions of interest the network focuses on when predicting `cancer` versus `normal` cases.

<p align="center">
  <img src="assets/gradcam_heatmap.jpg" width="800" alt="Grad-CAM Heatmap">
</p>

---

## 💻 Code Snippets for Evaluation & Visualization

### Evaluating on the Test Split

To evaluate your trained weights on the test dataset split via Python:

```python
from ultralytics import YOLO

# Load your best trained model weights
model = YOLO("results/weights/best.pt")

# Run evaluation on the 'test' split
metrics = model.val(
    data="dataset/data.yaml", 
    split="test",
    imgsz=640,
    batch=16
)

print(f"Test Box mAP50: {metrics.box.map50:.4f}")
print(f"Test Box mAP50-95: {metrics.box.map:.4f}")
```


## 👤 Author

* **Md Kawsar Mahmud**
* Feel free to open an issue or submit pull requests for suggestions and collaborations!
