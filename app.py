import cv2
import numpy as np
from tensorflow import keras
from function import mp_hands, mediapipe_detection, draw_styled_landmarks, extract_keypoints

# Load trained model
# NOTE: we rebuild the architecture from model.json (a stable, portable
# format) and load only the raw weight numbers from model.weights.h5.
# We deliberately avoid load_model() on a full .h5 file because Keras
# sometimes serializes internal details (like initializer parameters)
# that only match one specific Keras version - rebuilding from json +
# weights-only sidesteps that entirely and works across Keras versions.
with open("model.json") as f:
    _architecture = f.read()
model = keras.models.model_from_json(_architecture)
model.load_weights("model.weights.h5")

actions = np.array(['A', 'B', 'C'])
sequence_length = 30
sequence = []

cap = cv2.VideoCapture(0)

WINDOW_NAME = "Sign Detection"
cv2.namedWindow(WINDOW_NAME)

try:
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw active box
            cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)

            # Region of interest - this is where hand keypoints get detected
            roi = frame[40:400, 0:300]

            # Run mediapipe on the ROI and pull out the 63 hand keypoints
            image, results = mediapipe_detection(roi, hands)
            draw_styled_landmarks(image, results)
            frame[40:400, 0:300] = image

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]

            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

                action = actions[np.argmax(res)]
                confidence = res[np.argmax(res)] * 100

                if confidence > 50:
                    text = f"{action} ({confidence:.2f}%)"
                    color = (0, 255, 0)
                else:
                    text = "No sign detected"
                    color = (0, 0, 255)

                cv2.putText(frame, text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            color,
                            2)

            cv2.putText(frame, "Press 'q' to quit",
                        (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or Esc
                break

            # Also exit if the window was closed with the X button
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Windows sometimes needs this to actually close the window

