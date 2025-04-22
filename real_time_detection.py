import cv2
import numpy as np
import tensorflow as tf
import pickle
import os

# Load the trained model and label encoder
model = tf.keras.models.load_model('combined_lstm_model.h5')
with open('label_map.npy', 'rb') as f:
    label_map = np.load(f, allow_pickle=True).item()

# Setup
cap = cv2.VideoCapture(0)
SEQUENCE_LENGTH = 30
sequence = []

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Preprocess image for static gestures (alphabet)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))  # Resize to fixed size
    img = img.flatten().reshape(1, -1)  # Flatten for input to model

    # Preprocess dynamic gesture data (motion)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)

    if len(sequence) == SEQUENCE_LENGTH:
        prediction = model.predict(np.array(sequence).reshape(1, SEQUENCE_LENGTH, 42))
        predicted_label = np.argmax(prediction)
        label = list(label_map.keys())[predicted_label]
        cv2.putText(frame, f"Predicted: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
