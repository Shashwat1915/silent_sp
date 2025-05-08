# lip_movement_tracking.py
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time

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

# Output file
DATASET_FILE = "dataset/lip_dataset.json"
os.makedirs("dataset", exist_ok=True)

# Load existing dataset or create new one
if os.path.exists(DATASET_FILE):
    try:
        with open(DATASET_FILE, "r") as f:
            dataset = json.load(f)
    except json.JSONDecodeError:
        print("⚠️ Error loading JSON, creating a new dataset.")
        dataset = {}
else:
    dataset = {}

# Webcam
cap = cv2.VideoCapture(0)
recording = []
start_time = None
record_duration = 5  # seconds

print("🚀 Starting webcam. You will record for 5 seconds.")

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
            recording.append(lips)

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

# Ask user for label
label = input("✅ Enter the label for this sequence (e.g., hello): ").strip().lower()
if label:
    if label not in dataset:
        dataset[label] = []
    dataset[label].extend(recording)

    with open(DATASET_FILE, "w") as f:
        json.dump(dataset, f)

    print(f"✅ Saved {len(recording)} frames under label: {label}")
else:
    print("⚠️ No label entered. Data discarded.")
