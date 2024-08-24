from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp

app = Flask(__name__)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

hand_position = {"x": 0.5, "y": 0.5}  

def generate_frames():
    global hand_position
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Convert the image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = hands.process(frame_rgb)

            # Convert the image back to BGR for OpenCV
            frame_rgb.flags.writeable = True
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Draw landmarks and track hand position
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get the x and y coordinates of the wrist (landmark 0)
                    wrist_landmark = hand_landmarks.landmark[0]
                    hand_position = {"x": wrist_landmark.x, "y": wrist_landmark.y}  # Update hand position

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame_bgr)
            frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hand_position')
def hand_position_api():
    # Send the hand position as a JSON response
    return jsonify(hand_position)

if __name__ == '__main__':
    app.run(debug=True)
