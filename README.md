# Study Guard 👁️

A web-based focus monitoring app that uses your webcam for eye and face tracking to help you stay productive while studying.

## Features

- **Eye Tracking**: Detects when eyes are closed for too long (sleeping)
- **Face Direction**: Alerts if face is turned away from screen
- **Gaze Detection**: Warns if eyes are looking away from center
- **Focus Score**: Real-time calculation of attentiveness percentage
- **Web UI**: Clean, responsive web interface with live video feed
- **Real-time Updates**: Instant status updates via WebSocket
- **Alarm System**: Browser-based audio alerts when distracted
- **Cross-Platform**: Works on any device with a web browser and webcam

## Requirements

- Python 3.10+
- Webcam-enabled device
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Virtual environment (venv)

## Installation

1. Clone or download the code
2. Set up virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install flask flask-socketio opencv-python mediapipe pygame numpy
   ```

3. Alarm sound: Generated automatically using pygame and numpy (no external files needed)

## Usage

Run the web app:
```bash
source venv/bin/activate
python app.py
```

Open your browser and go to: `http://127.0.0.1:5000`

- Click "Start Monitoring"
- Allow camera access when prompted
- Position yourself in front of the webcam
- Watch your focus score and status in real-time
- Get alerted with visual and audio cues when distracted

## Web Interface Features

- **Live Video Feed**: Real-time webcam stream with face tracking overlay
- **Status Dashboard**: Current monitoring state and alerts
- **Focus Metrics**: Eyes, face, and gaze status indicators
- **Focus Score**: Percentage of attentive time
- **Control Buttons**: Start/stop monitoring with visual feedback
- **Responsive Design**: Works on desktop and mobile devices

## Customization

- Adjust detection thresholds in `app.py`:
  - `EYE_CLOSED_TIME`: Seconds before eye-closed alarm (default: 2.5)
  - `FACE_AWAY_TIME`: Seconds before face-away alarm (default: 3.0)
  - `LOOK_AWAY_TIME`: Seconds before look-away alarm (default: 2.0)
- Replace `static/alarm.wav` with your preferred alarm sound
- Modify CSS in `templates/index.html` for custom styling

## Architecture

- **Backend**: Flask + SocketIO for real-time communication
- **Computer Vision**: MediaPipe for accurate face landmark detection
- **WebRTC**: Direct webcam access through browser
- **Real-time Updates**: WebSocket events for instant UI updates

## Deployment

For production deployment:

1. Use a WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 app:app
   ```

2. Set up HTTPS for secure camera access
3. Configure firewall for the chosen port

## Browser Compatibility

- ✅ Chrome/Chromium (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge
- ⚠️ Mobile browsers (may have camera limitations)

## Notes

- Requires camera permission in browser
- Works offline once loaded
- Uses AI vision for privacy-focused local processing
- Fully customizable and open-source
- No data sent to external servers

Stay focused and productive! 🎯
# ATTENTION-ai
