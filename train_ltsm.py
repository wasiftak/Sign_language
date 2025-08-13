import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json

with open("lstm_label_map.json", "r") as f:
    label_map = json.load(f)
from data_preprocess import X_train, X_val, y_train, y_val
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(128, return_sequences=False, input_shape=(15, 126)),
    Dense(64, activation='relu'),
    Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=16)
# ...existing code...
model.save(r"C:\Users\WASIF TAK\OneDrive\Desktop\ai_project\lstm_sign_model.h5")
# ...existing code...