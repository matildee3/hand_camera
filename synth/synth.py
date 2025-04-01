from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import math
from flask_cors import CORS
import sounddevice as sd
from pythonosc.udp_client import SimpleUDPClient

app = Flask(__name__)
CORS(app)

# OSC Setup
OSC_IP = "127.0.0.1"
OSC_PORT = 7400
osc_client = SimpleUDPClient(OSC_IP, OSC_PORT)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Global variables
hand_position = {"x": 0.5, "y": 0.5, "distance": 0.5, "scale": 1.0}
hand_detected = False

# # Numeric Note Scale
# numeric_notes = {
#     0.0: 110,   # Lowest frequency
#     0.2: 130.81,
#     0.4: 146.83,
#     0.6: 164.81,
#     0.8: 196,   # Highest frequency
#     1.0: 220    # Additional top frequency
# }

def calculate_hand_size(hand_landmarks):
    """Calculate the relative size of the hand in the frame."""
    x_landmarks = [landmark.x for landmark in hand_landmarks.landmark]
    y_landmarks = [landmark.y for landmark in hand_landmarks.landmark]
    
    width = max(x_landmarks) - min(x_landmarks)
    height = max(y_landmarks) - min(y_landmarks)
    
    return width * height

def generate_frames():
    global hand_position, hand_detected
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = hands.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Calculate hand size and normalize
                    hand_size = calculate_hand_size(hand_landmarks)
                    normalized_hand_size = min(max(hand_size * 10, 0.1), 1.0)
                    
                    # wrist = hand_landmarks.landmark[0]
                    # hand_position = {
                    #     "x": wrist.x, 
                    #     "y": wrist.y, 
                    #     "distance": normalized_hand_size,
                    #     "scale": 0.5 + (normalized_hand_size * 0.5)  # Scale between 0.5 and 1.0
                    # }

                    wrist = hand_landmarks.landmark[0]  # Wrist
                    middle_mcp = hand_landmarks.landmark[9]  # Middle finger MCP joint

                    # Average the Y position between wrist and middle MCP to better center the hand
                    center_y = (wrist.y + middle_mcp.y) / 2

                    hand_position = {
                        "x": max(0.0, min(1.0, wrist.x )),  # Ensure it stays within [0,1] range
                        "y": max(0.0, min(1.0, center_y -0.1)),
                        "distance": normalized_hand_size,
                        "scale": 0.5 + (normalized_hand_size * 0.5)
                    }


            else:
                hand_detected = False
            
            ret, buffer = cv2.imencode('.jpg', frame_bgr)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# # Audio callback
# def audio_callback(outdata, frames, time, status):
#     global phase, hand_detected, hand_position
#     if status:
#         print(status)
    
#     t = (np.arange(frames) + phase) / sample_rate
    
#     if hand_detected:
#         x = hand_position["x"]
#         y = hand_position["y"]
#         distance = hand_position["distance"]
        
#         # Map x to numeric notes
#         note_keys = list(numeric_notes.keys())
#         note_index = min(max(int(x * len(note_keys)), 0), len(note_keys) - 1)
#         base_note_value = note_keys[note_index]
#         base_frequency = numeric_notes[base_note_value]
        
#         # More subtle octave variation
#         # Use interpolation between base and 1.5x frequency based on hand size
#         octave_interpolation = 1 + (distance - 0.5)  # ranges from 1 to 1.5
#         frequency = base_frequency * octave_interpolation
        
#         # Volume mapped to y-axis (top of screen = high volume, bottom = low volume)
#         # Invert y so that top of screen is high volume
#         volume = max(0.05, 1 - y)
        
#         # Send OSC messages
#         try:
#             print(f"Sending OSC - Note Value: {base_note_value}, Frequency: {frequency}, Volume: {volume}")
#             osc_client.send_message("/note", float(base_note_value))
#             osc_client.send_message("/frequency", float(frequency))
#             osc_client.send_message("/volume", float(volume))
#         except Exception as e:
#             print(f"OSC sending error: {e}")

#         wave = np.sin(2 * np.pi * frequency * t)
#         envelope = np.ones(frames)
        
#         attack_samples = int(0.005 * sample_rate)
#         release_samples = int(0.005 * sample_rate)
        
#         if frames > attack_samples:
#             envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
#         if frames > release_samples:
#             envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        
#         wave *= envelope * volume
#         phase = (phase + frames) % sample_rate
#     else:
#         wave = np.zeros(frames)
#         phase = 0.0
    
#     outdata[:] = wave.reshape(-1, 1)



# Modify audio callback to remove actual sound generation
def audio_callback(outdata, frames, time, status):
    global hand_detected, hand_position
    
    # Fill output with silence
    outdata[:] = np.zeros((frames, 1), dtype=np.float32)
    
    if hand_detected:
        x = hand_position["x"]
        y = hand_position["y"]
        distance = hand_position["distance"]
        
        # # Map x to numeric notes
        # note_keys = list(numeric_notes.keys())
        # note_index = min(max(int(x * len(note_keys)), 0), len(note_keys) - 1)
        # base_note_value = note_keys[note_index]
        # base_frequency = numeric_notes[base_note_value]
        #base_note_value = max(0, 1 - x)
        #base_frequency = 
        # More subtle octave variation
        base_note_value = x
        
        # Map base note value to a frequency range (for example, between 100 Hz and 1000 Hz)
        base_frequency = base_note_value 
        #print("base note ", base_note_value)
        


        octave_interpolation = 1 + (distance - 0.5)
        frequency = base_frequency * octave_interpolation
        
        # Volume mapped to y-axis (top of screen = high volume, bottom = low volume)
        volume = max(0.05, 1 - y)
        
        # Send OSC messages
        try:
            #print(f"Sending OSC - Note Value: {base_note_value}, Frequency: {frequency}, Volume: {volume}")
            osc_client.send_message("/note", float(base_note_value))
            osc_client.send_message("/frequency", float(frequency))
            osc_client.send_message("/volume", float(volume))
        except Exception as e:
            print(f"OSC sending error: {e}")


# Sound synthesis setup
sample_rate = 44100
buffer_size = 1024
phase = 0.0

# Start audio stream
stream = sd.OutputStream(
    samplerate=sample_rate, 
    channels=1, 
    callback=audio_callback, 
    blocksize=buffer_size,
    dtype='float32'
)
stream.start()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hand_position')
def hand_position_api():
    return jsonify(hand_position)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)