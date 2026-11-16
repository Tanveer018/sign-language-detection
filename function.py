import cv2
import numpy as np
import os
import mediapipe as mp

# Standard mediapipe hands solution (works with mediapipe==0.10.9)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                    # Image is no longer writeable
    results = model.process(image)                   # Make prediction
    image.flags.writeable = True                      # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)   # COLOR CONVERSION RGB 2 BGR
    return image, results


def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )


def extract_keypoints(results):
    """Always returns a fixed-length (63,) array so shapes never break
    training or real-time prediction, even when no hand is detected."""
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = np.array(
            [[res.x, res.y, res.z] for res in hand_landmarks.landmark]
        ).flatten()
        return keypoints
    return np.zeros(21 * 3)


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

actions = np.array(['A', 'B', 'C'])

no_sequences = 30

sequence_length = 30
