# Sign Detection Project — Fixed

## What was actually broken (and verified fixed, end-to-end, in a real environment)

1. **`function.py`** used `mp_drawing` / `mp_drawing_styles` without ever
   importing them, and had a broken `mp_hands` fallback — this crashed
   `data.py`. Also, `extract_keypoints()` returned `None` (instead of
   zeros) whenever no hand was visible, which crashes training/prediction
   the moment a frame has no hand in it.
2. **`app.py`** had mediapipe stripped out and was feeding the model a
   sequence of all zeros — it never actually looked at your hand. Fixed
   to run real mediapipe hand detection every frame and feed real
   keypoints to the model.
3. **`data.py`** read one static image and copied it into all 30 "frames"
   of a sequence — not real motion data. Fixed to capture live, frame by
   frame, from your webcam.
4. **`collectdata.py`** crashed immediately because it called
   `os.listdir()` on folders that didn't exist yet. Fixed with
   `os.makedirs`. Note: this script is for a *different*, image-based
   approach — your actual model is an LSTM trained on hand keypoints, not
   raw images, so this script isn't required for `app.py` to work.
5. **`trainmodel.py`** mixed `keras.*` and `tensorflow.keras.*` imports,
   a common source of version conflicts. Standardized on
   `tensorflow.keras`.
6. **The model file itself couldn't be loaded reliably.** The original
   `model.h5` was saved with an old Keras version and contained an LSTM
   argument (`time_major`) that current Keras rejects. My first fix
   (re-saving a full `model_fixed.h5`) turned out to have the *same kind*
   of problem in reverse — Keras's full-model save format embeds
   internal details (like extra initializer parameters) that can differ
   between Keras patch versions, so a file saved by one Keras version can
   fail to load in another.

   The robust fix: `app.py` now rebuilds the model architecture fresh
   from **`model.json`** (a simple, stable format with no version-specific
   internals) and loads only the raw trained numbers from
   **`model.weights.h5`**. This avoids Keras's fragile full-model
   serialization entirely and is what actually works across Keras
   versions. This was verified by installing the exact dependency
   versions below into a brand new, empty environment and successfully
   loading the model and running a prediction.

## Verified working dependency versions

```
opencv-python==4.10.0.84
mediapipe==0.10.21
tensorflow==2.19.0
numpy==1.26.4
scikit-learn==1.7.2
```

These exact versions were installed together from scratch and confirmed
to have no dependency conflicts, and the full model-loading + mediapipe
hand-detection + prediction pipeline was tested successfully with them.
I can't test the actual webcam feed in my environment (no camera
available there), so that part depends on your machine's camera working
correctly — everything else has been verified.

## Step-by-step: how to run this

### 1. Install Python packages
Open a terminal in the project folder (with your virtual environment
activated) and run:
```
pip install -r requirements.txt
```

### 2. Run real-time sign detection (uses your already-trained model)
```
python app.py
```
- A window opens showing your webcam feed with a white detection box.
- Put your hand inside the box.
- It takes ~30 frames (about 1 second) to fill the buffer, then it will
  show the predicted letter (A, B, or C) with a confidence percentage.
- Press **q** to quit.

### 3. (Optional) Collect your own training data instead of using the provided model
```
python data.py
```
- Walks through actions A, B, C, 30 sequences each, 30 frames per
  sequence, capturing live from your webcam and saving keypoints as
  `.npy` files under `MP_Data/`.
- Get your hand in position before each "STARTING COLLECTION" pause.
- Press **q** at any point to stop early.

### 4. (Optional) Retrain the model on your new data
```
python trainmodel.py
```
- Trains for 200 epochs and saves the result as `model.h5`.
- Afterward, run `python fix_model.py` once to regenerate
  `model.weights.h5` from the new `model.h5`, so `app.py` picks up your
  retrained weights.

### 5. (Optional) `collectdata.py`
This is a leftover from an alternate, image-based approach and isn't
needed for the current keypoint-based model — skip it unless you're
building a separate image classifier.

## Files
- `app.py` — real-time webcam sign detection (run this to see it work)
- `function.py` — shared mediapipe/keypoint helper functions
- `data.py` — live webcam data collection (keypoints)
- `trainmodel.py` — trains the LSTM model on collected keypoints
- `fix_model.py` — regenerates `model.weights.h5` after retraining
- `collectdata.py` — optional, alternate image-collection script (unused by app.py)
- `model.h5` — your original trained weights (kept for reference / retraining script)
- `model.json` — model architecture (used by app.py and fix_model.py)
- `model.weights.h5` — raw trained weights only, used by app.py (this is the file actually loaded at runtime)
