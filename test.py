import cv2, numpy as np, json
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque

# Load model and labels
model = load_model("lstm_sign_model2.h5")
with open("lstm_label_map2.json") as f:
    label_map = json.load(f)
labels = {v: k for k, v in label_map.items()}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
buffer = deque(maxlen=15)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmarks = []

    if result.multi_hand_landmarks:
        # Sort by x position for consistency (left to right)
        hand_list = sorted(result.multi_hand_landmarks,
                           key=lambda hand: hand.landmark[0].x)

        for hand in hand_list[:2]:  # take max 2 hands
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == 63:
            landmarks.extend([0.0] * 63)  # pad second hand if missing
    else:
        landmarks = [0.0] * 126  # no hand detected

    buffer.append(landmarks)

    # Predict when buffer full
    if len(buffer) == 15:
        sequence = np.array(buffer)[-15:]
        prediction = model.predict(np.expand_dims(sequence, axis=0), verbose=0)
        pred_class = np.argmax(prediction)
        pred_label = labels[pred_class]
        confidence = prediction[0][pred_class]

        if confidence > 0.7:
            cv2.putText(frame, f"{pred_label} ({confidence:.2f})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw hands
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("LSTM Dual-Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
