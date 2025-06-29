# import cv2
# import os
# import numpy as np
# import mediapipe as mp
# import time

# # ==== YOU SHOULD CHANGE THIS PART ====
# actions = ['Z_asl']  # <-- your labels here
# DATA_PATH = os.path.join('MP_Data_LSTM')  # <-- dataset save folder
# no_sequences = 100  # <-- sequences per label
# sequence_length = 15  # <-- frames per sequence
# # =====================================

# # Setup Mediapipe
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
# mp_draw = mp.solutions.drawing_utils

# cap = cv2.VideoCapture(0)

# for action in actions:
#     for sequence in range(no_sequences):
#         print(f"\n🕒 Ready to record: '{action}' - Sample [{sequence}]")
#         frame_buffer = []

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame = cv2.flip(frame, 1)
#             image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(image_rgb)

#             # Show landmarks for reference
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             cv2.putText(frame, f"Action: {action} | Sample: {sequence}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.putText(frame, "Press 'S' to start, 'Q' to quit", (10, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#             cv2.imshow('Sequence Collector (Dual Hand)', frame)

#             key = cv2.waitKey(1)
#             if key & 0xFF == ord('s'):
#                 print("▶️ Recording started...")
#                 break
#             elif key & 0xFF == ord('q'):
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 exit()

#         # Start capturing sequence
#         while len(frame_buffer) < sequence_length:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame = cv2.flip(frame, 1)
#             image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(image_rgb)

#             keypoints = []

#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#                     for lm in hand_landmarks.landmark:
#                         keypoints.extend([lm.x, lm.y, lm.z])
#                 if len(keypoints) < 126:
#                     keypoints.extend([0] * (126 - len(keypoints)))
#             else:
#                 # If no hands detected, skip the frame (don't append junk)
#                 continue

#             frame_buffer.append(keypoints)
#             cv2.putText(frame, f'Capturing {len(frame_buffer)}/{sequence_length}', (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#             cv2.imshow('Sequence Collector (Dual Hand)', frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         # Save sequence
#         folder_path = os.path.join(DATA_PATH, action)
#         os.makedirs(folder_path, exist_ok=True)
#         np.save(os.path.join(folder_path, f'{action}_{sequence}'), frame_buffer)
#         print("✅ Sequence saved!")

# cap.release()
# cv2.destroyAllWindows()

import cv2
import os
import numpy as np
import mediapipe as mp
import time

# ==== YOU SHOULD CHANGE THIS PART ====
actions = ['A_isl']  # <-- your labels here
DATA_PATH = os.path.join('MP_Data_LSTM')  # <-- dataset save folder
no_sequences = 100  # <-- sequences per label
sequence_length = 15  # <-- frames per sequence
# =====================================

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

for action in actions:
    for sequence in range(no_sequences):
        print(f"\n🕒 Ready to record: '{action}' - Sample [{sequence}]")
        frame_buffer = []

        # Show camera feed + 1-second countdown
        start_time = time.time()
        while time.time() - start_time < 1:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            countdown = 1 - int(time.time() - start_time)
            cv2.putText(frame, f"Action: {action} | Sample: {sequence}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Recording in: {countdown}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow('Sequence Collector (Auto Start)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        print("▶️ Recording started...")

        # Start capturing sequence
        while len(frame_buffer) < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            keypoints = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for lm in hand_landmarks.landmark:
                        keypoints.extend([lm.x, lm.y, lm.z])
                if len(keypoints) < 126:
                    keypoints.extend([0] * (126 - len(keypoints)))
            else:
                continue  # Skip frame if no hand

            frame_buffer.append(keypoints)
            cv2.putText(frame, f'Capturing {len(frame_buffer)}/{sequence_length}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow('Sequence Collector (Auto Start)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save sequence
        folder_path = os.path.join(DATA_PATH, action)
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, f'{action}_{sequence}'), frame_buffer)
        print("✅ Sequence saved!")

cap.release()
cv2.destroyAllWindows()
