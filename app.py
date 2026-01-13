from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import time
import pygame
import numpy as np
import base64
import threading
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize sound
pygame.mixer.init()

# Generate alarm sound programmatically
def generate_beep(frequency=800, duration=0.5, volume=0.8):
    """Generate a simple beep sound using numpy and pygame"""
    import tempfile
    import wave

    sample_rate = 44100
    samples = int(sample_rate * duration)

    # Generate sine wave
    t = np.linspace(0, duration, samples, False)
    wave_data = np.sin(frequency * 2 * np.pi * t) * volume

    # Convert to 16-bit PCM
    wave_data = (wave_data * 32767).astype(np.int16)

    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_filename = temp_file.name
        with wave.open(temp_file, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(wave_data.tobytes())

    # Load the temporary file into pygame
    sound = pygame.mixer.Sound(temp_filename)

    # Clean up temp file
    os.unlink(temp_filename)

    return sound

try:
    alarm_sound = generate_beep(frequency=800, duration=0.5, volume=0.8)
    print("Generated alarm beep sound successfully.")
except Exception as e:
    print(f"Warning: Could not generate alarm sound: {e}")
    alarm_sound = None

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

EYE_CLOSED_TIME = 2.5  # seconds
FACE_AWAY_TIME = 3.0   # seconds
LOOK_AWAY_TIME = 2.0   # seconds

eye_closed_start = None
face_away_start = None
look_away_start = None

# Focus tracking
total_time = 0
attentive_time = 0
eye_closure_count = 0
look_away_count = 0
start_time = 0

monitoring = False

def eye_aspect_ratio(landmarks, eye_indices):
    pts = [landmarks[i] for i in eye_indices]
    vertical = abs(pts[1].y - pts[5].y)
    horizontal = abs(pts[0].x - pts[3].x)
    return vertical / horizontal

def get_face_direction(landmarks):
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[362]
    eye_center_x = (left_eye.x + right_eye.x) / 2
    eye_center_y = (left_eye.y + right_eye.y) / 2
    yaw = nose.x - eye_center_x
    pitch = nose.y - eye_center_y
    return yaw, pitch

def get_eye_gaze(landmarks, eye_indices):
    eye_pts = [landmarks[i] for i in eye_indices]
    eye_center_x = sum(p.x for p in eye_pts) / len(eye_pts)
    eye_center_y = sum(p.y for p in eye_pts) / len(eye_pts)
    return eye_center_x, eye_center_y

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def generate_frames():
    global eye_closed_start, face_away_start, look_away_start, total_time, attentive_time, eye_closure_count, look_away_count, monitoring

    while True:  # Always keep the stream alive
        ret, frame = cap.read()
        if not ret:
            # Create a placeholder frame when camera is not available
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if monitoring:
            current_time = time.time()

            # Only process frames when monitoring is active
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            eye_closed = False
            face_away = False
            look_away = False
            status = "Monitoring..."

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark

                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2

                if ear < 0.20:
                    eye_closed = True
                    if eye_closed_start is None:
                        eye_closed_start = time.time()
                        eye_closure_count += 1
                    elif time.time() - eye_closed_start > EYE_CLOSED_TIME:
                        status = "ALARM: Eyes Closed!"
                        if alarm_sound:
                            alarm_sound.play(-1)
                else:
                    eye_closed_start = None
                    if alarm_sound:
                        alarm_sound.stop()

                yaw, pitch = get_face_direction(landmarks)
                if abs(yaw) > 0.1 or abs(pitch) > 0.1:
                    face_away = True
                    if face_away_start is None:
                        face_away_start = time.time()
                    elif time.time() - face_away_start > FACE_AWAY_TIME:
                        status = "ALARM: Face Away!"
                        if alarm_sound:
                            alarm_sound.play(-1)
                else:
                    face_away_start = None

                left_gaze_x, _ = get_eye_gaze(landmarks, LEFT_EYE)
                right_gaze_x, _ = get_eye_gaze(landmarks, RIGHT_EYE)
                gaze_x = (left_gaze_x + right_gaze_x) / 2
                if abs(gaze_x - 0.5) > 0.2:
                    look_away = True
                    if look_away_start is None:
                        look_away_start = time.time()
                        look_away_count += 1
                    elif time.time() - look_away_start > LOOK_AWAY_TIME:
                        status = "ALARM: Looking Away!"
                        if alarm_sound:
                            alarm_sound.play(-1)
                else:
                    look_away_start = None

                if not (eye_closed or face_away or look_away):
                    attentive_time += 0.1

                focus_score = int((attentive_time / total_time) * 100) if total_time > 0 else 100

                eye_status = "Closed" if eye_closed else "Open"
                face_status = "Away" if face_away else "Facing Screen"
                gaze_status = "Away" if look_away else "Center"

                # Emit status update
                socketio.emit('update', {
                    'status': status,
                    'focus_score': focus_score,
                    'eye_status': eye_status,
                    'face_status': face_status,
                    'gaze_status': gaze_status,
                    'alarm': 'ALARM' in status
                })
            else:
                if monitoring:
                    status = "ALARM: No Face!"
                    if alarm_sound:
                        alarm_sound.play(-1)
                    socketio.emit('update', {
                        'status': status,
                        'focus_score': int((attentive_time / total_time) * 100) if total_time > 0 else 100,
                        'eye_status': "Unknown",
                        'face_status': "Not Detected",
                        'gaze_status': "Unknown",
                        'alarm': True
                    })
        else:
            # When not monitoring, show a waiting message
            cv2.putText(frame, "Click 'Start Monitoring'", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "to begin focus tracking", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Encode frame for web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Small delay to prevent excessive CPU usage
        time.sleep(0.1)

@socketio.on('start_monitoring')
def start_monitoring():
    global monitoring, total_time, attentive_time, eye_closed_start, face_away_start, look_away_start, start_time
    monitoring = True
    # Reset timers when starting new monitoring session
    total_time = 0
    attentive_time = 0
    eye_closed_start = None
    face_away_start = None
    look_away_start = None
    start_time = time.time()  # Reset start time for accurate timing
    emit('status', {'monitoring': True})
    # Emit initial status with 0% focus score
    socketio.emit('update', {
        'status': 'Monitoring...',
        'focus_score': 0,
        'eye_status': 'Open',
        'face_status': 'Facing Screen',
        'gaze_status': 'Center',
        'alarm': False
    })

@socketio.on('stop_monitoring')
def stop_monitoring():
    global monitoring
    monitoring = False
    if alarm_sound:
        alarm_sound.stop()
    emit('status', {'monitoring': False})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)
