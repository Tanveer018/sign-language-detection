"""
You normally don't need to run this - model.weights.h5 is already included.

Run this ONLY if you retrain and get a new model.h5, and want to
regenerate a fresh model.weights.h5 for app.py to use.

Why weights-only instead of the full model.h5? Keras's full-model save
format embeds internal serialization details that can differ between
Keras versions (we hit this directly - a model saved with one Keras
patch version failed to load with another, e.g. errors mentioning
'time_major' or 'input_axes'). Rebuilding the architecture fresh from
model.json (a stable, simple format) and loading only the raw weight
numbers sidesteps that problem entirely.

Usage:
    python fix_model.py
"""

from tensorflow import keras

with open("model.json") as f:
    architecture = f.read()

model = keras.models.model_from_json(architecture)
model.load_weights("model.h5")
model.save_weights("model.weights.h5")

print("Done! Saved model.weights.h5 - this is the file app.py loads.")
