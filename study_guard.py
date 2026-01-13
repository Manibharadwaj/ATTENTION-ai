import cv2
import mediapipe as mp
import time
import pygame
import numpy as np
import rumps
import os
from threading import Thread

# Initialize sound
pygame.mixer.init()
alarm_file = "alarm.wav"
alarm_sound = None
if os.path.exists(alarm_file):
    try:
        alarm_sound = pygame.mixer.Sound(alarm_file)
    except:
        print("Warning: Could not load alarm.wav. No alarm sound.")
        alarm_sound = None
else:
    print("Warning: alarm.wav not found. No alarm sound.")

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

def eye_aspect_ratio(landmarks, eye_indices):
    pts = [landmarks[i] for i in eye_indices]
    vertical = abs(pts[1].y - pts[5].y)
    horizontal = abs(pts[0].x - pts[3].x)
    return vertical / horizontal

def get_face_direction(landmarks):
    # Simple face direction based on nose position relative to eyes
    nose = landmarks[1]  # Nose tip
    left_eye = landmarks[33]
    right_eye = landmarks[362]
    eye_center_x = (left_eye.x + right_eye.x) / 2
    eye_center_y = (left_eye.y + right_eye.y) / 2

    # Yaw: horizontal deviation
    yaw = nose.x - eye_center_x
    # Pitch: vertical deviation
    pitch = nose.y - eye_center_y

    return yaw, pitch

def get_eye_gaze(landmarks, eye_indices):
    # Simplified gaze: check if pupils are towards center
    # For simplicity, use eye center
    eye_pts = [landmarks[i] for i in eye_indices]
    eye_center_x = sum(p.x for p in eye_pts) / len(eye_pts)
    eye_center_y = sum(p.y for p in eye_pts) / len(eye_pts)
    return eye_center_x, eye_center_y

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

class StudyGuardApp(rumps.App):
    def __init__(self):
        super(StudyGuardApp, self).__init__("👁️", quit_button=None)
        self.status_item = rumps.MenuItem("Status: Initializing...")
        self.focus_item = rumps.MenuItem("Focus Score: 100%")
        self.eye_item = rumps.MenuItem("Eyes: Open")
        self.face_item = rumps.MenuItem("Face: Facing Screen")
        self.gaze_item = rumps.MenuItem("Gaze: Center")
        self.start_item = rumps.MenuItem("Start Monitoring", callback=self.start_monitoring)
        self.stop_item = rumps.MenuItem("Stop Monitoring", callback=self.stop_monitoring)
        self.stop_item.state = False  # Disabled initially
        self.menu = [
            self.status_item,
            self.focus_item,
            self.eye_item,
            self.face_item,
            self.gaze_item,
            None,
            self.start_item,
            self.stop_item,
            None,
            rumps.MenuItem("Quit", callback=self.quit_app)
        ]
        self.monitoring = False
        self.thread = None

    def start_monitoring(self, _):
        self.monitoring = True
        self.start_item.state = True  # Disabled
        self.stop_item.state = False  # Enabled
        self.status_item.title = "Status: Monitoring Active"
        self.thread = Thread(target=self.monitor)
        self.thread.start()

    def stop_monitoring(self, _):
        self.monitoring = False
        if self.thread:
            self.thread.join()
        self.start_item.state = False
        self.stop_item.state = True
        self.status_item.title = "Status: Monitoring Stopped"
        if alarm_sound:
            alarm_sound.stop()

    def quit_app(self, _):
        self.stop_monitoring(_)
        rumps.quit_application()

    def update_menu(self, eye_status, face_status, gaze_status, focus_score):
        self.focus_item.title = f"Focus Score: {focus_score}%"
        self.eye_item.title = f"Eyes: {eye_status}"
        self.face_item.title = f"Face: {face_status}"
        self.gaze_item.title = f"Gaze: {gaze_status}"

    def monitor(self):
        global eye_closed_start, face_away_start, look_away_start, total_time, attentive_time, eye_closure_count, look_away_count

        start_time = time.time()

        while self.monitoring and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            total_time = current_time - start_time

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            eye_closed = False
            face_away = False
            look_away = False

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark

                # Eye aspect ratio
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2

                if ear < 0.20:  # eye closed
                    eye_closed = True
                    if eye_closed_start is None:
                        eye_closed_start = current_time
                        eye_closure_count += 1
                    elif current_time - eye_closed_start > EYE_CLOSED_TIME:
                        if alarm_sound:
                            alarm_sound.play(-1)  # Loop alarm
                        self.status_item.title = "Status: ALARM - Eyes Closed!"
                else:
                    eye_closed_start = None
                    if alarm_sound:
                        alarm_sound.stop()

                # Face direction
                yaw, pitch = get_face_direction(landmarks)
                if abs(yaw) > 0.1 or abs(pitch) > 0.1:  # Face not facing screen
                    face_away = True
                    if face_away_start is None:
                        face_away_start = current_time
                    elif current_time - face_away_start > FACE_AWAY_TIME:
                        if alarm_sound:
                            alarm_sound.play(-1)
                        self.status_item.title = "Status: ALARM - Face Away!"
                else:
                    face_away_start = None

                # Eye gaze
                left_gaze_x, _ = get_eye_gaze(landmarks, LEFT_EYE)
                right_gaze_x, _ = get_eye_gaze(landmarks, RIGHT_EYE)
                gaze_x = (left_gaze_x + right_gaze_x) / 2
                if abs(gaze_x - 0.5) > 0.2:  # Eyes not looking at center
                    look_away = True
                    if look_away_start is None:
                        look_away_start = current_time
                        look_away_count += 1
                    elif current_time - look_away_start > LOOK_AWAY_TIME:
                        if alarm_sound:
                            alarm_sound.play(-1)
                        self.status_item.title = "Status: ALARM - Looking Away!"
                else:
                    look_away_start = None

                # Attentive if not eye closed, face away, or look away
                if not (eye_closed or face_away or look_away):
                    attentive_time += 0.1  # Assuming 10 FPS
            else:
                # No face detected
                if alarm_sound:
                    alarm_sound.play(-1)
                self.status_item.title = "Status: ALARM - No Face!"

            # Calculate focus score
            if total_time > 0:
                focus_score = int((attentive_time / total_time) * 100)
            else:
                focus_score = 100

            # Update menu
            eye_status = "Closed" if eye_closed else "Open"
            face_status = "Away" if face_away else "Facing Screen"
            gaze_status = "Away" if look_away else "Center"
            self.update_menu(eye_status, face_status, gaze_status, focus_score)

            # Show frame
            cv2.imshow("Study Guard 👁️", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):  # 100ms delay
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    StudyGuardApp().run()
