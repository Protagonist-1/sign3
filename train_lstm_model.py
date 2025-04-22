import os
import pickle
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directories
DATA_DIR = './MP_Data'
SEQUENCE_LENGTH = 30  # Number of frames per sequence
IMG_SIZE = 64  # Resize images to 64x64

# Data & Labels for both Letters and Phrases
data = []
labels = []
label_map = {}  # To store label mapping

# 1️⃣ Process Letter Data (A-Z) for Left & Right Hand
print("Processing letter images (A-Z)...")
letter_base_index = 0

for hand_type in ['left_hand', 'right_hand']:  # Loop through both hands
    HAND_DIR = os.path.join(DATA_DIR, hand_type)

    for label in os.listdir(HAND_DIR):  # 0-25 folders for A-Z
        label_path = os.path.join(HAND_DIR, label)
        label_int = int(label)

        if label_int not in label_map:
            label_map[label_int] = chr(65 + label_int)  # Map 0 → A, 1 → B, ..., 25 → Z

        for img_path in os.listdir(label_path):
            data_aux = []
            x_, y_ = [], []

            img = cv2.imread(os.path.join(label_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x_.append(hand_landmarks.landmark[i].x)
                        y_.append(hand_landmarks.landmark[i].y)

                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                        data_aux.append(hand_landmarks.landmark[i].y - min(y_))

                data.append(data_aux)
                labels.append(label_int)

# 2️⃣ Process Phrase Data (e.g., 'hello', 'thankyou')
print("Processing phrase images...")
phrase_base_index = 26  # Letters go from 0-25, so phrases start at 26
PHRASE_DIR = os.path.join(DATA_DIR, 'phrases')

for idx, phrase in enumerate(os.listdir(PHRASE_DIR)):  
    phrase_path = os.path.join(PHRASE_DIR, phrase)
    phrase_label = phrase_base_index + idx  # Assign unique number to phrases

    label_map[phrase_label] = phrase  # e.g., 26 -> 'hello', 27 -> 'thank_you'

    for img_path in os.listdir(phrase_path):
        data_aux = []
        x_, y_ = [], []

        img = cv2.imread(os.path.join(phrase_path, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            data.append(data_aux)
            labels.append(phrase_label)

# 3️⃣ Save Dataset with both Letters & Phrases
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels, 'label_map': label_map}, f)
print("Dataset created successfully! ✅")

# 4️⃣ Load Dataset and Train Models

# Load the dataset
try:
    with open('data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print("Error: data.pickle not found. Run the dataset creation first.")
    exit()

# Check data consistency and pad shorter samples
max_length = max(len(item) for item in data_dict['data'])
padded_data = [np.pad(item, (0, max_length - len(item)), 'constant') for item in data_dict['data']]
data = np.asarray(padded_data)

# Encode labels numerically
labels = np.asarray(data_dict['labels'])
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)  # Convert text labels (phrases) to numbers

# Print label distribution
unique_labels = set(labels)
print(f"Unique labels: {unique_labels} (Total classes: {len(unique_labels)})")

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

# Evaluate Random Forest Model
y_predict = rf_model.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}% ✅")

# Save Random Forest Model & Label Encoder
with open('rf_model.p', 'wb') as f:
    pickle.dump({'model': rf_model, 'label_encoder': label_encoder}, f)
print("✅ Random Forest Model & Label Encoder saved successfully!")

# Train LSTM Model for Sequential Data

# Load data and labels for LSTM model (sequence-based gestures)
X, y = [], []
label_map = {label: idx for idx, label in enumerate(os.listdir(DATA_DIR))}

for label in label_map:
    label_folder = os.path.join(DATA_DIR, label)
    for file in os.listdir(label_folder):
        data = np.load(os.path.join(label_folder, file))
        if data.shape == (SEQUENCE_LENGTH, 42):
            X.append(data)
            y.append(label_map[label])

X = np.array(X)
y = to_categorical(y).astype(int)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 42)))
lstm_model.add(LSTM(64, return_sequences=False, activation='relu'))
lstm_model.add(Dense(64, activation='relu'))
lstm_model.add(Dense(32, activation='relu'))
lstm_model.add(Dense(len(label_map), activation='softmax'))

# Compile and train
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Save the LSTM model & label map
lstm_model.save('lstm_model.h5')
np.save('label_map.npy', label_map)
print("✅ LSTM Model trained and saved.")
