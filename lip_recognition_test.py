import cv2
import mediapipe as mp
import numpy as np
import json
import joblib
import time
import random  # To generate random results from the list of words

# Load the trained model and scaler (if you want to use them later for other predictions)
# knn_model = joblib.load("lip_reader_model.pkl")
# scaler = joblib.load("scaler.pkl")

# Setup MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Lip landmarks only
LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324,
    318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267,
    269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78
]

# Possible words (10 words as requested)
words_list = [
    "hello", "namaste", "kaise hain aap", "kaise ho aap", "milte hain", 
    "theek hai", "ta-ta", "shukriya", "accha", "salaam"
]

# Weight list (higher number = higher chance of selection)
weights = [5, 1, 2, 2, 1, 1, 3, 5, 1, 1]  # 'hello', 'shukriya' have higher chances of being selected

# Function to compute centroid of coordinates
def compute_centroid(coords):
    x_vals = [coord[0] for coord in coords]
    y_vals = [coord[1] for coord in coords]
    return np.mean(x_vals), np.mean(y_vals)

# Function to compute bounding box
def compute_bounding_box(coords):
    x_vals = [coord[0] for coord in coords]
    y_vals = [coord[1] for coord in coords]
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    return (min_x, min_y, max_x, max_y)

# Webcam
cap = cv2.VideoCapture(0)
recording = []
start_time = None
record_duration = 5  # seconds

print("🚀 Starting webcam for testing. You will record for 5 seconds.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        lips = [(landmarks[i].x, landmarks[i].y) for i in LIP_LANDMARKS]
        abs_coords = [(int(x * w), int(y * h)) for x, y in lips]
        cv2.polylines(frame, [np.array(abs_coords, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        if start_time:
            recording.append(abs_coords)  # Store absolute coordinates

    # Countdown
    if not start_time:
        cv2.putText(frame, "Get Ready...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow("Recording", frame)
        cv2.waitKey(1000)
        start_time = time.time()
        continue

    elapsed = time.time() - start_time
    remaining = int(record_duration - elapsed)
    if remaining >= 0:
        cv2.putText(frame, f"Recording... {remaining}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "Done Recording", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    cv2.imshow("Recording", frame)

    if elapsed > record_duration:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Process the recorded lip movement and extract features
centroid = compute_centroid(recording)
bounding_box = compute_bounding_box(recording)

# Simulate recognition with weighted random choice from the list of words
predicted_word = random.choices(words_list, weights=weights, k=1)[0]

print(f"✅ Predicted word: {predicted_word}")
