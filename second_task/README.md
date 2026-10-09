# Speech Emotion Recognition (SER) using PyTorch & 2D CNN

An end-to-end Deep Learning pipeline for 7-class Speech Emotion Recognition (SER) trained on the Toronto Emotional Speech Set (TESS). The system converts raw audio signals into Mel-Frequency Cepstral Coefficients (MFCCs) and classifies emotional states using a custom 2D Convolutional Neural Network (CNN) in PyTorch, achieving **99.52% test accuracy**.

---

## Key Highlights

- **Framework**: PyTorch & Librosa
- **Dataset**: Toronto Emotional Speech Set (TESS)
- **Classes (7)**: Angry, Disgust, Fear, Happy, Neutral, Pleasant, Sad
- **Architecture**: Custom 2D CNN with Batch Normalization, Dropout, and Kaiming Initialization
- **Performance**: 99.52% Test Accuracy with minimal cross-entropy loss

---

## Dataset Overview

The dataset consists of audio recordings categorized into 7 emotion classes across two female speakers (OAF and YAF).

- **Total Samples**: 2,800 audio files (.wav format)
- **Data Splits**:
  - Training Set (70%): 1,960 samples
  - Validation Set (15%): 420 samples
  - Test Set (15%): 420 samples
- **Stratification**: Applied across all splits to preserve identical emotion class distributions.

---

## Audio Preprocessing & Feature Extraction

1. **Signal Processing**: Raw time-domain waveforms are converted to 20-band Mel-Frequency Cepstral Coefficients (MFCCs) using `librosa`.
2. **Dimension Standardization**: All extracted feature matrices are padded or truncated to a fixed time-frame width of 174, resulting in uniform 2D feature tensors of shape `(20, 174)`.
3. **Data Pipeline**: Custom PyTorch `SpeechDataset` class manages tensor formatting, dynamic batch loading, and memory placement (CPU/GPU).

---

## Model Architecture (`SpeechNN`)

The network uses stacked 2D convolutional operations designed to capture spatial-temporal patterns across audio spectrograms:

- **ConvBlock 1**: Conv2d (1 -> 32 channels, 3x3 kernel) + BatchNorm2d + ReLU + MaxPool2d (2x2)
- **ConvBlock 2**: Conv2d (32 -> 64 channels, 3x3 kernel) + BatchNorm2d + ReLU + MaxPool2d (2x2)
- **ConvBlock 3**: Conv2d (64 -> 128 channels, 3x3 kernel) + BatchNorm2d + ReLU + MaxPool2d (2x2)
- **Fully Connected Layers**: Dense layers with Dropout for regularization and a 7-unit Logits output layer.
- **Weight Initialization**: Kaiming Normal (He) initialization applied across convolutional layers for stable gradient flow.

---

## Training Configuration

- **Optimizer**: `AdamW` (learning rate = 0.0005, weight decay = 0.01)
- **Loss Function**: `CrossEntropyLoss`
- **Learning Rate Scheduler**: `ReduceLROnPlateau` (factor = 0.5, patience = 3)
- **Gradient Clipping**: Value clipping threshold set to `1.0`
- **Metrics Tracking**: `torchmetrics` for per-epoch accuracy logging

---

## Evaluation & Results

- **Validation Accuracy**: High stability with loss convergence below 0.02.
- **Test Set Accuracy**: **99.52%**
- **Test Set Loss**: **0.0130**
- **Inference Pipeline**: Includes a single-audio inference helper function (`predict_single_audio`) returning the predicted emotion along with softmax probability confidence scores.
