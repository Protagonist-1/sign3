import os
import cv2
import numpy as np
import mediapipe as mp

# Setup
SEQUENCE_LENGTH = 30
SEQUENCES_PER_CLASS = 10
DATA_PATH = './MP_Data'
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# Define classes for both static (letters) and motion (phrases)
LETTERS_CLASSES = [str(i) for i in range(26)]  # A-Z
PHRASES = ['hello', 'thank_you', 'i_love_you', 'help', 'sorry']  # List of phrases

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def extract_keypoints(results, flip=False):
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
            if flip:
                hand = [[1 - x, y] for x, y in hand]  # Flip along X-axis
            keypoints.append(np.array(hand).flatten())
    if len(keypoints) == 0:
        return np.zeros(42)
    return keypoints[0][:42]  # Return keypoints of the first hand detected

# Start video capture
cap = cv2.VideoCapture(0)

# Collect static images (letters A-Z)
mode = input("Do you want to collect letters (A-Z) or phrases? (Enter 'letters' or 'phrases'): ").strip().lower()

if mode == 'letters':
    DATA_DIR = os.path.join(DATA_PATH, 'letters')
    classes = LETTERS_CLASSES
elif mode == 'phrases':
    DATA_DIR = os.path.join(DATA_PATH, 'phrases')
    classes = PHRASES
else:
    print("Invalid choice! Exiting...")
    cap.release()
    cv2.destroyAllWindows()
    exit()

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Collect data for static images (A-Z) or motion sequences (phrases)
for label in classes:
    class_path = os.path.join(DATA_DIR, label)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    print(f'Collecting data for "{label}"...')

    # Wait for user to be ready
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Get Ready for {label}! Press "Q" to start', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    if label in LETTERS_CLASSES:  # Static image collection (A-Z)
        dataset_size = 50  # Number of images per class
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(class_path, f'{counter}.jpg'), frame)
            counter += 1

    elif label in PHRASES:  # Motion sequence collection (phrases)
        for seq in range(SEQUENCES_PER_CLASS):
            sequence = []
            flipped_sequence = []
            print(f'Collecting phrase "{label}", Sequence {seq+1}/{SEQUENCES_PER_CLASS}')

            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    continue

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Extract normal and flipped keypoints
                keypoints = extract_keypoints(results)
                flipped_keypoints = extract_keypoints(results, flip=True)

                sequence.append(keypoints)
                flipped_sequence.append(flipped_keypoints)

                # Draw hand landmarks (optional)
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.putText(image_bgr, f'Phrase: {label} | Frame: {frame_num+1}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('Collecting', image_bgr)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # Save sequence (original and flipped)
            save_path = os.path.join(DATA_PATH, 'phrases', label)
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, f'{label}_{seq}'), sequence)
            np.save(os.path.join(save_path, f'{label}_{seq}_flipped'), flipped_sequence)

cap.release()
cv2.destroyAllWindows()
print("Data collection completed! ✅")
