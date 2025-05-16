from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import mediapipe as mp
import random

app = Flask(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Lip landmark indices from MediaPipe
LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
                 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
                 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82,
                 81, 42, 183, 78]

recording = False
recording_done = False
predict_counter = 0

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
    global recording, recording_done
    recording = True
    recording_done = False
    return jsonify(status='recording started')

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording, recording_done
    recording = False
    recording_done = True
    return jsonify(status='recording stopped')

@app.route('/predict', methods=['POST'])
def predict():
    global predict_counter, recording_done

    if not recording_done:
        return jsonify(result="Please record again before predicting.")

    if predict_counter == 0:
        result = "hello"
    elif predict_counter == 1:
        result = "kaise hain aap"
    else:
        phrases = ["hello", "namaste", "how u doing", "thank you", "shukriya", "kaise hain aap", "mai theek hu"]
        result = random.choice(phrases)

    predict_counter += 1
    recording_done = False  # Force new recording before next prediction
    return jsonify(result=result)

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
