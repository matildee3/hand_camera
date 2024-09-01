from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Load the image you want to display
overlay_image = cv2.imread('ball.png', cv2.IMREAD_UNCHANGED)  # Ensure this is a PNG with transparency

def gen_frames():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1) as hands:
        
        while True:
            success, frame = cap.read()
            if not success:
                continue
            
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get landmarks for thumb tip and index finger tip
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    # Convert normalized coordinates to pixel coordinates
                    h, w, _ = frame.shape
                    thumb_tip_x, thumb_tip_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    index_tip_x, index_tip_y = int(index_tip.x * w), int(index_tip.y * h)

                    # Calculate the distance between thumb and index finger
                    distance = np.sqrt((thumb_tip_x - index_tip_x) ** 2 + (thumb_tip_y - index_tip_y) ** 2)

                    # Adjust the threshold for showing the image (more strict to disappear)
                    if distance > 60:  # Adjusted threshold for showing the image
                        # Adjust the scale to be smaller so the image fits closely between fingers
                        scale = int(distance * 1.2)  # Scale the image slightly less than the distance
                        overlay_resized = cv2.resize(overlay_image, (scale, scale))

                        # Calculate position between thumb and index
                        center_x = (thumb_tip_x + index_tip_x) // 2
                        center_y = (thumb_tip_y + index_tip_y) // 2

                        # Overlay the image on the frame
                        overlay_x = center_x - scale // 2
                        overlay_y = center_y - scale // 2

                        # Ensure the overlay doesn't go outside the frame
                        if overlay_x >= 0 and overlay_y >= 0 and overlay_x + scale <= w and overlay_y + scale <= h:
                            # Add the overlay image onto the frame
                            alpha_s = overlay_resized[:, :, 3] / 255.0
                            alpha_l = 1.0 - alpha_s

                            for c in range(0, 3):
                                frame[overlay_y:overlay_y+scale, overlay_x:overlay_x+scale, c] = (
                                    alpha_s * overlay_resized[:, :, c] +
                                    alpha_l * frame[overlay_y:overlay_y+scale, overlay_x:overlay_x+scale, c]
                                )

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
