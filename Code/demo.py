import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model/drowsiness_sequence_gru.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Facial landmark indices
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
TOP_LIP_IDX = 13
BOTTOM_LIP_IDX = 14
LEFT_MOUTH_IDX = 78
RIGHT_MOUTH_IDX = 308

# EAR calculation
def compute_ear(landmarks, w, h):
    def ear_for(idxs):
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in idxs]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (A + B) / (2.0 * C + 1e-8)
    return (ear_for(LEFT_EYE_IDXS) + ear_for(RIGHT_EYE_IDXS)) / 2.0

# MAR calculation
def compute_mar(landmarks, w, h):
    pt_top   = np.array((landmarks[TOP_LIP_IDX].x * w, landmarks[TOP_LIP_IDX].y * h))
    pt_bot   = np.array((landmarks[BOTTOM_LIP_IDX].x * w, landmarks[BOTTOM_LIP_IDX].y * h))
    pt_left  = np.array((landmarks[LEFT_MOUTH_IDX].x * w, landmarks[LEFT_MOUTH_IDX].y * h))
    pt_right = np.array((landmarks[RIGHT_MOUTH_IDX].x * w, landmarks[RIGHT_MOUTH_IDX].y * h))
    vert = np.linalg.norm(pt_top - pt_bot)
    horz = np.linalg.norm(pt_left - pt_right)
    return vert / (horz + 1e-8)

# Initialize feature sequence deque
SEQ_LEN = 60
feature_sequence = deque(maxlen=SEQ_LEN)

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to height 640 first, keeping aspect ratio
    h_target = 640
    scale = h_target / frame.shape[0]
    resized_w = int(frame.shape[1] * scale)
    frame = cv2.resize(frame, (resized_w, h_target))

    # Crop width to center 360px (for 9:16 aspect ratio)
    w_target = 360
    start_x = max((resized_w - w_target) // 2, 0)
    frame = frame[:, start_x:start_x + w_target]

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        lm = results.multi_face_landmarks[0].landmark

        # Compute EAR and MAR
        ear = compute_ear(lm, w, h)
        mar = compute_mar(lm, w, h)

        feature_sequence.append([ear, mar])

        if len(feature_sequence) == SEQ_LEN:
            input_data = np.array(feature_sequence, dtype=np.float32).reshape(1, SEQ_LEN, 2)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            drowsiness_prob = output[0][0]

            status = "Drowsy" if drowsiness_prob > 0.7 else "Alert"
            color = (0, 0, 255) if status == "Drowsy" else (0, 255, 0)

            cv2.putText(frame, f"Status: {status} ({drowsiness_prob:.2f})", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()