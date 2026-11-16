import os
import cv2

# NOTE: This script is for an alternate, image-based data collection approach.
# Your current model.h5 / model.json is an LSTM trained on hand *keypoints*
# (see function.py / data.py / trainmodel.py) and does NOT use these images.
# Only use this script if you plan to train a separate image-classifier model.

cap = cv2.VideoCapture(0)
directory = 'Image/'
letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Create all needed folders up front so os.listdir() never crashes
for letter in letters:
    os.makedirs(os.path.join(directory, letter), exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count = {
        letter.lower(): len(os.listdir(os.path.join(directory, letter)))
        for letter in letters
    }

    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
    cv2.imshow("data", frame)

    roi = frame[40:400, 0:300]
    cv2.imshow("ROI", roi)

    interrupt = cv2.waitKey(10) & 0xFF

    if interrupt == ord('q'):
        break

    if 0 <= interrupt < 256:
        key = chr(interrupt)
        if key.isalpha() and key.upper() in letters:
            letter = key.upper()
            save_path = os.path.join(directory, letter, f"{count[key]}.png")
            cv2.imwrite(save_path, roi)

cap.release()
cv2.destroyAllWindows()
