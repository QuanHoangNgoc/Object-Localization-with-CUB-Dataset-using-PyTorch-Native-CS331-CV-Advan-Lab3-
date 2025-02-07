# Object Localization with CUB Dataset using PyTorch

## Overview
This project focuses on object localization using the **Caltech-UCSD Birds 200 (CUB)** dataset, a widely used benchmark for fine-grained visual recognition tasks. The goal is to develop a **PyTorch-based** solution that integrates **bounding box localization** and **classification**, leveraging a **ResNet-based** architecture with custom loss functions.

## Dataset and Preprocessing
### Steps:
- Cloned the **CUB dataset** from Kaggle and structured it for smooth integration.
- **Preprocessing Workflow:**
  - Loaded and merged relevant files into a **consolidated DataFrame**.
  - Created a **custom PyTorch Dataset class** to serve data efficiently.
  - Formatted data into `train_dataloader` and `val_dataloader` for streamlined training and validation.

## Model Design and Loss Functions
### Architecture:
- Utilizes a **ResNet18 backbone** for feature extraction.
- Splits into two **fully connected branches**:
  1. **Bounding Box Localization**: Predicts `(x_cen, y_cen, w, h)`, normalized to `[0, 1]`.
  2. **Classification**: Outputs class probabilities.

### Custom Loss Function:
- **Composite Loss:**
  - **Square-root Mean Squared Error (sqrtMSE)** for bounding box localization.
  - **Cross-Entropy (CE)** for classification.
- Inspired by **YOLOv1**, we use the square-root format for `w` and `h` to improve stability for small object predictions.
- Both losses are weighted with scaling factors to balance localization and classification performance.

## Training Configuration
- **Data Type:** `float32`, deployed on **GPU** for enhanced computation speed.
- **Optimizer:** **Adam** for efficient gradient updates.
- **Metrics Tracked:**
  - **Training & Validation Loss**
  - **Intersection-over-Union (IOU)** for bounding box accuracy
  - **Classification Accuracy**

## Evaluation Metrics
- **IOU Metric:** Assesses bounding box prediction accuracy.
- **Classification Accuracy:** Measures correctness in label predictions.

## Experimental Results
After **30 epochs**, the model achieved:
- **Average IOU:** **92%**
- **Classification Accuracy:** **15%**

## Efficiency and Resource Requirements
- **Training Time:** ~15 minutes
- **GPU Memory Usage:** ~1GB
- Provided **visualizations** of:
  - **Training history** (loss reduction, IOU improvement, classification accuracy)
  - **Test predictions** with bounding boxes

## Conclusion
This project presents an **efficient and straightforward** approach to object localization, balancing computational efficiency with effective bounding box predictions. Despite its simplicity, the model achieves strong **localization performance** with **minimal resource usage**, demonstrating its potential for practical applications.

## Repository Link
- **GitHub:** [Object Localization with CUB Dataset using PyTorch](https://github.com/QuanHoangNgoc/Object-Localization-with-CUB-Dataset-using-PyTorch-Native-CS331-CV-Advan-Lab3)

