import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from tensorflow.keras.models import load_model
import os

# === 1. Load the data ===
data = np.load("MP_Data_LSTM")
X = data[:, :-28]  # features (30 * 126 = 3780)
y = data[:, -28:]  # one-hot labels (28 classes)

# === 2. Reshape X to (samples, 30, 126) ===
X = X.reshape(-1, 30, 126)

# === 3. Split into train and test sets ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 4. Save test set for reuse ===
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

# === 5. Load the label map ===
with open("lstm_label_map2.json", "r") as f:
    label_map = json.load(f)
reverse_label_map = {v: k for k, v in label_map.items()}

# === 6. Verify 28 classes ===
labels = [reverse_label_map[i] for i in range(28)]
if len(labels) != 28:
    raise ValueError(f"Expected 28 labels, got {len(labels)}")

# === 7. Load the model ===
model = load_model("lstm_sign_model2.h5")
print("Loaded model.")

# === 8. Predict on test data ===
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# === 9. Print evaluation metrics ===
accuracy = accuracy_score(y_true, y_pred_classes)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred_classes, average="weighted", zero_division=0
)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# === 10. Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (LSTM Model)")

# Ensure output folder exists
os.makedirs("lstm_outputs", exist_ok=True)
plt.tight_layout()
plt.savefig("lstm_outputs/confusion_matrix.png")
plt.close()
print("Confusion matrix saved to lstm_outputs/confusion_matrix.png")
