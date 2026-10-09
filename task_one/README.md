# 🧠 MNIST Digit Classification using PyTorch (98.9% Accuracy)

An end-to-end Computer Vision pipeline implementing a Custom Convolutional Neural Network (CNN) built with **PyTorch** to classify handwritten digits from the classic **MNIST dataset**.

---

## 📌 Project Overview

This repository demonstrates a clean, production-ready Deep Learning workflow for image classification. It covers everything from dataset ingestion, explicit data splitting, batching, weight initialization, training loop implementation with real-time metrics tracking, to comprehensive evaluation using loss curves and confusion matrices.

### Key Highlights:
* **Framework:** PyTorch & TorchVision
* **Custom Architecture:** Dual-Block CNN with Kaiming (He) Weight Initialization
* **Optimization:** AdamW Optimizer with Cross-Entropy Loss
* **Evaluation Metrics:** Accuracy tracking via TorchMetrics, Scikit-Learn Confusion Matrix, and Seaborn visual heatmaps
* **Test Accuracy:** **98.92%** on unseen test data

---

## 🏗️ Model Architecture

The `MNISTNN` architecture consists of two spatial feature extraction blocks followed by a dense classification head:

1. **Convolutional Block 1:**
   * `Conv2d` (1 → 32 channels, $3 \times 3$ kernel, padding=1)
   * `ReLU` Activation
   * `MaxPool2d` ($2 \times 2$, stride=2) — Downsamples to $14 \times 14$
   
2. **Convolutional Block 2:**
   * `Conv2d` (32 → 64 channels, $3 \times 3$ kernel, padding=1)
   * `ReLU` Activation
   * `MaxPool2d` ($2 \times 2$, stride=2) — Downsamples to $7 \times 7$

3. **Classifier Head:**
   * `Flatten` ($64 \times 7 \times 7 \rightarrow 3136$)
   * Linear Layer (3136 → 10 classes)

> **Initialization:** All convolutional layers use **Kaiming Normal Initialization** (`kaiming_normal_`) to prevent vanishing/exploding gradients with explicit zero-bias initialization.

---

## 📊 Dataset & Preprocessing

* **Training Set:** 50,000 samples (split from standard 60k)
* **Validation Set:** 10,000 samples (used for tracking metrics per epoch)
* **Test Set:** 10,000 unseen samples
* **Transformations:** Scaled via `transforms.ToTensor()` to normal range $[0.0, 1.0]$.
* **Batch Size:** 32 across all loaders (`train_loader` shuffled).

---

## 🚀 Results & Performance

| Split | Loss | Accuracy |
| :--- | :--- | :--- |
| **Training** | `0.0150` | `99.56%` |
| **Validation** | `0.0626` | `98.66%` |
| **Test Set** | **`0.0526`** | **`98.92%`** |

### Saved Artifacts:
* Trained model weights are serialized and saved locally as `mnist_model.pth`.

---

## 📂 Repository Structure

```text
.
├── notebooks/
│   └── mnist_cnn_pytorch.ipynb   # Main Jupyter Notebook with detailed Markdown
├── weights/
│   └── mnist_model.pth           # Saved PyTorch model state_dict
├── README.md                     # Project documentation
└── requirements.txt              # Dependencies
