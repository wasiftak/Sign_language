Real-Time Sign Language Recognition with LSTM
💡 Introduction
This project enables real-time recognition of dynamic sign language gestures using an LSTM neural network. By analyzing sequences of hand landmarks (extracted via MediaPipe), the model classifies isolated signs from both American (ASL) and Indian (ISL) alphabets. Designed for low-latency deployment, it bridges communication gaps without relying on sensors or static images.

📄 Custom Dataset (LSTM-Specific)
Unlike CNN-based approaches (which use pre-existing Kaggle datasets), this LSTM model trains on a custom-collected dataset:

Data Source: Real-time webcam captures with MediaPipe landmark extraction.

Classes: 52 (A–Z in ASL + A–Z in ISL).

Samples:

5,000+ labeled sequences (15 frames each).

126 features/frame: 3D coordinates (x, y, z) for 21 landmarks × 2 hands.

Variations: Speed, lighting, and hand positioning.

Preprocessing: Zero-padding for missing hands, normalized coordinates.

(Note: The CNN path uses Kaggle’s ASL/ISL datasets, but this LSTM pipeline is fully custom.)

⚙️ How It Works
🎯 Model Architecture
Input: 15-frame sliding window of hand landmarks (15×126 tensor).

LSTM Layer: 128 units to capture temporal dynamics.

Output: Softmax over 52 classes.

Training: 10 epochs (80:20 split), Adam optimizer.

⚡ Real-Time Pipeline
Landmark Extraction: MediaPipe detects hands and outputs 21 landmarks per hand.

Sequence Buffering: Frames are aggregated into 15-frame sequences.

Prediction: LSTM classifies the gesture if confidence > 90%.

📊 Results
Metric	Score
Accuracy	96.48%
Precision	0.9697
Recall	0.9648
F1-Score	0.9638
AUC-ROC	0.98
Key Findings:

High accuracy but struggles with two-handed signs (e.g., ASL ‘N’ vs. ISL ‘K’).

Minimal overfitting (validation accuracy plateaus at ~96.5%).

https://media/image5.png
Figure: Misclassifications occur mainly between similar gestures.

🚀 Try It Yourself
Prerequisites
bash
pip install mediapipe tensorflow opencv-python  
Run Real-Time Demo
bash
python lstm_inference.py --model_path ./models/lstm_landmark.h5  
Train from Scratch
Record your own sequences:

python
python collect_landmarks.py --output_dir ./custom_data  
Train:

bash
python train_lstm.py --data_path ./custom_data/sequences.npy  
