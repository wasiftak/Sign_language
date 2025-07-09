# 🤟 Real-Time Sign Language Recognition with LSTM

## 💡 Introduction
This project enables **real-time recognition of dynamic sign language gestures** using an LSTM-based neural network.  
By analyzing sequences of hand landmarks (extracted with MediaPipe), the model can classify isolated signs from both **American Sign Language (ASL)** and **Indian Sign Language (ISL)** alphabets.  
Designed for **low-latency, sensor-free deployments**, this system bridges communication gaps without relying on static images or gloves.

---

## 📄 Dataset
- **Name**: Custom In-house Sign Landmark Dataset
- **Source**: Real-time webcam recordings using MediaPipe landmark extraction
- **Classes**: 52 (A–Z in ASL + A–Z in ISL)
- **Samples**:  
  - 5,000+ labeled sequences
  - Each sequence contains 15 frames
  - Each frame has 126 features (3D coordinates from 21 landmarks × 2 hands)

---

## ⚙️ How It Works

### 🎯 Model Architecture
- **Input**: 15-frame sequence (shape: 15 × 126)
- **Core**: LSTM layer with 128 units to capture temporal dynamics
- **Output**: Softmax over 52 classes
- **Training**:  
  - 10 epochs  
  - 80:20 split (train:validation)  
  - Optimizer: Adam

---

### ⚡ Real-Time Pipeline
1. **Hand Landmark Extraction**: MediaPipe detects 21 landmarks per hand in each frame.
2. **Sequence Buffering**: Frames are collected into 15-frame sliding windows.
3. **Prediction**: LSTM classifies the sign if confidence > 90%.

---

## 📊 Results

| Metric    | Score  |
|------------|---------|
| Accuracy | 96.48% |
| Precision | 0.9697 |
| Recall    | 0.9648 |
| F1-Score | 0.9638 |
| AUC-ROC | 0.98  |

### ✅ Key Observations
- High accuracy on most single-handed signs
- Some confusion in two-handed or visually similar gestures (e.g., ASL 'N' vs ISL 'K')
- Minimal overfitting (validation accuracy stabilizes ~96.5%)

---

## 🚀 Quick Start

### 🔧 Install Requirements
```bash
pip install mediapipe tensorflow opencv-python
