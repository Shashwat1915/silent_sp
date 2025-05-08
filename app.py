from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import random

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
                 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
                 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82,
                 81, 42, 183, 78]

recording = False
cap = cv2.VideoCapture(0)

def gen_frames():
    global recording
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            for i in LIP_LANDMARKS:
                x = int(landmarks[i].x * frame.shape[1])
                y = int(landmarks[i].y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Resize to 50% for smaller display
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording
    recording = True
    return jsonify(status='recording started')

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording
    recording = False
    return jsonify(status='recording stopped')

@app.route('/predict', methods=['POST'])
def predict():
    phrases = ["hello", "namaste", "how u doing", "thank you", "shukriya", "kaise hain aap", "mai theek hu"]
    return jsonify(result=random.choice(phrases))

if __name__ == '__main__':
    app.run(debug=True, port=5001)
