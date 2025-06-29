import os
import numpy as np
from sklearn.model_selection import train_test_split
import json

# === PATH TO YOUR DATASET ===
DATA_PATH = "MP_Data_LSTM"

X, y = [], []
label_map = {}

# === Load sequences from each class ===
for idx, class_name in enumerate(os.listdir(DATA_PATH)):
    label_map[class_name] = idx
    class_path = os.path.join(DATA_PATH, class_name)
    
    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)
        try:
            sequence = np.load(file_path)
            if sequence.shape == (15, 126):
                X.append(sequence)
                y.append(idx)
            else:
                print(f"⚠️ Skipped {file} (shape {sequence.shape})")
        except Exception as e:
            print(f"❌ Failed to load {file}: {e}")

# === Convert to numpy arrays ===
X = np.array(X)
y = np.array(y)

# === Split into train/test ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

# === Save label map ===
with open("lstm_label_map2.json", "w") as f:
    json.dump(label_map, f)

# === Optional: Save the processed data ===
np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)

# === Print Summary ===
print("✅ Data preprocessing complete!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", label_map)
